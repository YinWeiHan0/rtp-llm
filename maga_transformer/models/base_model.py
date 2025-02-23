import os
import torch
from dataclasses import dataclass, field
from pydantic import BaseModel as PyBaseModel
from typing import Any, Dict, List, Optional, Tuple, Union, NamedTuple
from transformers import PreTrainedTokenizerBase

from maga_transformer.config.task_type import TaskType
from maga_transformer.ops.ft_op_base import FTOPBase
from maga_transformer.utils.weight_type import WEIGHT_TYPE
from maga_transformer.distribute.worker_info import g_parallel_info
from maga_transformer.config.generate_config import GenerateConfig
from maga_transformer.models.downstream_modules.custom_module import CustomModule
from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters
from maga_transformer.ops.comm.parallel_op import ParallelEmbedding, ParallelLinear
from maga_transformer.utils.multimodal_util import MultimodalInput

FT_DEFAULT_MAX_NEW_TOKENS = 2048

class EmbeddingOutput:
    text_embedding: torch.Tensor
    extra_input: Optional[torch.Tensor]

    def __init__(self, text_embedding: torch.Tensor, extra_input: Optional[List[torch.Tensor]]):
        self.text_embedding = text_embedding
        if extra_input:
            try:
                self.extra_input = torch.concat(extra_input)
                self.extra_input = torch.Tensor(self.extra_input.shape[1:])
            except:
                raise Exception("Extra input must have same shape except dim 0")
        else:
            self.extra_input = None

# single batch prompt input
class GenerateInput(PyBaseModel):
    request_id: int
    token_ids: torch.Tensor
    mm_inputs: List[MultimodalInput]
    generate_config: GenerateConfig
    tokenizer: Any = None # TODO: remove this
    prefix_length: int = 0
    token_type_ids: List[int] = []

    class Config:
        arbitrary_types_allowed = True

    @property
    def input_length(self):
        return self.token_ids.shape[-1]

    @property
    def prompt_length(self):
        return self.token_ids.shape[-1] - self.prefix_length

    def update_prefix(self, prefix_tokens: torch.Tensor):
        self.token_ids = torch.concat([prefix_tokens, self.token_ids], dim=0)
        self.prefix_length = prefix_tokens.nelement()

class AuxInfo(PyBaseModel):
    cost_time: float = 0
    iter_count: int = 0
    prefix_len: int = 0
    input_len: int = 0
    reuse_len: int = 0
    output_len: int = 0
    step_output_len: int = 0
    fallback_tokens: int = 0
    fallback_times: int = 0
    cum_log_probs: List[float] = []
    beam_responses: List[str] = []

class GenerateOutput(PyBaseModel):
    hidden_states: Optional[torch.Tensor] = None
    output_ids: Optional[torch.Tensor] = None
    input_ids: Optional[torch.Tensor] = None
    finished: bool = False
    aux_info: AuxInfo = AuxInfo()
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None

    class Config:
        arbitrary_types_allowed = True

class GenerateOutputs(PyBaseModel):
    generate_outputs: List[GenerateOutput] = []

class GenerateResponse(PyBaseModel):
    generate_outputs: GenerateOutputs
    generate_texts: List[str]

class GenerateContext(NamedTuple):
    inputs: Any
    input_embeds: Any
    attention_mask: Any
    pad_lengths: Any
    input_lengths: Any
    memory_length: Any
    sampler: Any
    batch_size: Any
    beam_width: Any
    max_input_length: Any
    finished: Any
    sequence_lengths: Any
    gen_length: Any
    cum_log_probs: Any
    extra_args: Any
    all_start_time: Any
    cache_indirection: Any
    output_token_ids: Any

class ModelConfig:
    def __init__(
            self,
            model_type: str = "",
            ckpt_path: str = "",
            tokenizer_path: str = "",
            weight_type: WEIGHT_TYPE = WEIGHT_TYPE.FP16,
            act_type: WEIGHT_TYPE = WEIGHT_TYPE.FP16,
            max_seq_len: int = 0,
            seq_size_per_block: int = 8,
            gen_num_per_circle: int = 1,
            ptuning_path: Optional[str] = None,
            lora_infos: Optional[Dict[str, str]] = None,
            ref_module: Optional[torch.nn.Module] = None,
            ref_dict: Dict[str, torch.Tensor] = {},
            sp_type: str = "",
        ):
        self.model_type: str = model_type
        self.ckpt_path: str = ckpt_path
        self.tokenizer_path: str = tokenizer_path
        self.weight_type: WEIGHT_TYPE = weight_type
        self.act_type: WEIGHT_TYPE = act_type
        self.max_seq_len: int = max_seq_len
        self.seq_size_per_block: int = seq_size_per_block
        self.gen_num_per_circle: int = gen_num_per_circle
        self.ptuning_path: Optional[str] = ptuning_path
        self.lora_infos: Optional[Dict[str, str]] = lora_infos
        self.ref_module: Optional[torch.nn.Module] = ref_module
        self.ref_dict: Dict[str, torch.Tensor] = ref_dict
        self.sp_type: str = sp_type

    @property
    def int8_mode(self):
        return True if self.weight_type == WEIGHT_TYPE.INT8 else False

    def add_ref_module(self, ref_module: Optional[torch.nn.Module]):
        self.ref_module = ref_module

    def add_ref_dict(self, ref_dict: Dict[str, torch.Tensor]):
        self.ref_dict = ref_dict

    def _replace(self, **kwargs: Any):
        for k, v in kwargs.items():
            if k in self.__dict__:
                self.__dict__[k] = v
        return self

class BaseModel(object):

    config: GptInitModelParameters
    vocab_size_padded: int
    device: str

    @classmethod
    def create_config(cls, model_config: ModelConfig) -> GptInitModelParameters:
        config: GptInitModelParameters = cls._create_config(model_config.ckpt_path)
        if config.hidden_size == 0:
            config.hidden_size = config.size_per_head * config.head_num
        config.update_common(
            ckpt_path=model_config.ckpt_path,
            tokenizer_path=model_config.tokenizer_path,
            int8_mode=model_config.int8_mode,
            data_type=model_config.act_type,
            max_seq_len=model_config.max_seq_len,
            seq_size_per_block=model_config.seq_size_per_block,
            tp_size=g_parallel_info.tp_size,
            gen_num_per_circle=model_config.gen_num_per_circle,
            lora_infos=model_config.lora_infos,
            ptuning_path=model_config.ptuning_path,
            ref_module=model_config.ref_module,
            ref_dict=model_config.ref_dict
        )
        return config

    @staticmethod
    def _create_config(ckpt_path: str) -> GptInitModelParameters:
        raise NotImplementedError()

    @classmethod
    def from_config(cls, config: Any) -> 'BaseModel':
        raise NotImplementedError()

    @classmethod
    def get_tokenizer(cls, config: GptInitModelParameters) -> PreTrainedTokenizerBase:
        raise NotImplementedError()

    def is_multimodal(self) -> bool:
        raise NotImplementedError()

    def __init__(self) -> None:
        self.weight = None
        self.word_embedding: Optional[ParallelEmbedding] = None
        self.prefix_encoder: Optional[torch.nn.Module] = None
        self.position_encoding: Optional[ParallelEmbedding] = None
        self.token_type_embeddings: Optional[ParallelEmbedding] = None
        self.pre_decoder_layernorm: Optional[torch.nn.Module] = None
        self.post_decoder_layernorm: Optional[torch.nn.Module] = None

        self.lm_head: Optional[ParallelLinear] = None
        self.config: GptInitModelParameters = None
        self.context_decoder: Optional[FTOPBase] = None
        self.decoder: Optional[FTOPBase] = None
        self.dynamic_decoder = None
        self.use_fp32_to_compute_logit = False
        self.linear_bias_slopes: Optional[torch.Tensor] = None

        self.medusa_head: Optional[torch.nn.ModuleList] = None

        self.prefix_tokens: Optional[torch.Tensor] = None
        self.tokenizer: Optional[PreTrainedTokenizerBase] = None
        self.max_input_buffer_len: int = 0

        self.task_type: TaskType = TaskType.LANGUAGE_MODEL
        self.custom_module: Optional[CustomModule] = None

        self.default_generate_config: GenerateConfig = GenerateConfig()
        self.device = g_parallel_info.device

    @property
    def dtype(self) -> Union[str, torch.dtype]:
        assert self.weight is not None
        return self.weight.dtype

    def dup_dim0_for_beam_search(self, t: torch.Tensor, beam_width: int) -> torch.Tensor:
        shape = list(t.shape)
        return t.unsqueeze(1).repeat([1, beam_width] + [1] * len(shape[1:])).reshape([-1] + shape[1:]).contiguous()

    def extend_context_combo_token_types(self, token_types: List[int]) -> List[int]:
        return []

    def extend_generate_combo_token_types(self, combo_tokens: List[int]) -> List[int]:
        return []

    def create_context_position_ids(self, input_lengths: Union[List[int], torch.Tensor]):
        return torch.concat([torch.arange(int(input_length), dtype=torch.int32) for input_length in input_lengths], dim=0)

    def create_context_decoder_mask(self, input_lengths: List[int]):
        batch_size = len(input_lengths)
        max_input_length = max(input_lengths)
        attention_mask = torch.ones(
            (max_input_length, max_input_length), dtype=torch.bool, device=self.device)
        if self.config.is_causal:
            attention_mask = attention_mask.tril()
        attention_mask = attention_mask.unsqueeze_(0).tile(batch_size, 1, 1).to(self.dtype)
        for b, input_length in enumerate(input_lengths):
            attention_mask[b, input_length:, ...] = 0
            if not self.config.is_causal:
                attention_mask[b, :, input_length: ]= 0
        return attention_mask

    @staticmethod
    def eval_model_size(config: GptInitModelParameters):
        return config.eval_model_size()

    @staticmethod
    def eval_model_param_count(config: GptInitModelParameters):
        return config.model_param_count
