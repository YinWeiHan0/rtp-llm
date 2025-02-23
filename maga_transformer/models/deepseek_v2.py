import torch
import unicodedata
import types
import functools
import math
import os
import json
from typing import List, Any

from maga_transformer.models.gpt import GPT
from maga_transformer.utils.model_weight import (
    W,
    ModelDeployWeightInfo,
    ModelWeightInfo,
    WeightInfo,
    CkptWeightInfo,
    identity,
    transpose,
    stack_,
    w_half1,
    w_half2,
    zeros,
    transpose_pad,
    multipy_identity
)
from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters
from maga_transformer.model_factory_register import register_model

def yarn_get_mscale(scale: float=1, mscale: float=1):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def kv_split1(
    ts: List[torch.Tensor], kv_lora_rank: int, nope_head_dim: int, v_head_dim: int
) -> torch.Tensor:
    k, _ = (
        ts[0]
        .transpose(0, 1)
        .reshape(kv_lora_rank, -1, nope_head_dim + v_head_dim)
        .split([nope_head_dim, v_head_dim], dim=-1)
    )
    k = k.reshape(kv_lora_rank, -1)
    return k.contiguous()


def kv_split2(
    ts: List[torch.Tensor], kv_lora_rank: int, nope_head_dim: int, v_head_dim: int
) -> torch.Tensor:
    _, v = (
        ts[0]
        .transpose(0, 1)
        .reshape(kv_lora_rank, -1, nope_head_dim + v_head_dim)
        .split([nope_head_dim, v_head_dim], dim=-1)
    )
    v = v.reshape(kv_lora_rank, -1)
    return v.contiguous()


def mla_pad_t(ts: List[torch.Tensor], head_num: int, nope_head_dim: int, rope_head_dim: int) -> torch.Tensor:
    t = ts[0]
    t = t.reshape(-1, head_num, nope_head_dim)
    z = torch.zeros(t.shape[0], head_num, rope_head_dim, device=t.device, dtype=t.dtype)
    t = torch.cat([t, z], dim=-1)
    t = t.reshape(-1, head_num * (nope_head_dim + rope_head_dim))
    return t.T.contiguous()


class DeepSeekV2Weight(ModelDeployWeightInfo):
    q_use_lora = False

    def __init__(self, config: GptInitModelParameters, tp_size: int, tp_rank: int):
        super().__init__(config, tp_size, tp_rank)

    def _process_meta(self, meta_dict, weight_keys):
        if "model.layers.0.self_attn.q_a_proj.weight" in weight_keys:
            self.q_use_lora = True

    def _get_hf_layer_weight_info(self, layer_id: int):
        layer_weights = [
            WeightInfo(W.pre_ln_gamma, [CkptWeightInfo('model.layers.{i}.input_layernorm.weight', identity)],
                       identity),
            WeightInfo(W.attn_o_w, [CkptWeightInfo('model.layers.{i}.self_attn.o_proj.weight', identity)],
                       functools.partial(mla_pad_t, head_num=self._head_num, nope_head_dim=self.nope_head_dim, rope_head_dim=self.rope_head_dim)),
            WeightInfo(W.post_ln_gamma, [CkptWeightInfo('model.layers.{i}.post_attention_layernorm.weight', identity)],
                       identity),
        ]
        mla_layer_weights = [
            WeightInfo(W.mla_kv_a_w, [CkptWeightInfo('model.layers.{i}.self_attn.kv_a_proj_with_mqa.weight', identity)],
                       functools.partial(w_half1, inter_size=self.kv_lora_rank)),
            WeightInfo(W.mla_k_rope_w, [CkptWeightInfo('model.layers.{i}.self_attn.kv_a_proj_with_mqa.weight', identity)],
                       functools.partial(w_half2, inter_size=self.kv_lora_rank)),
            WeightInfo(W.mla_k_nope_w, [CkptWeightInfo('model.layers.{i}.self_attn.kv_b_proj.weight', identity)],
                       functools.partial(kv_split1, kv_lora_rank=self.kv_lora_rank, nope_head_dim=self.nope_head_dim, v_head_dim=self.v_head_dim)),
            WeightInfo(W.mla_v_w, [CkptWeightInfo('model.layers.{i}.self_attn.kv_b_proj.weight', identity)],
                       functools.partial(kv_split2, kv_lora_rank=self.kv_lora_rank, nope_head_dim=self.nope_head_dim, v_head_dim=self.v_head_dim)),
            WeightInfo(W.mla_kv_a_ln_gamma, [CkptWeightInfo('model.layers.{i}.self_attn.kv_a_layernorm.weight', identity)],
                       identity),
        ]

        if self.q_use_lora:
            mla_layer_weights.extend([
                WeightInfo(W.mla_q_a_w, [CkptWeightInfo('model.layers.{i}.self_attn.q_a_proj.weight', identity)],
                        transpose),
                WeightInfo(W.mla_q_b_w, [CkptWeightInfo('model.layers.{i}.self_attn.q_b_proj.weight', identity)],
                        transpose),
                WeightInfo(W.mla_q_a_ln_gamma, [CkptWeightInfo('model.layers.{i}.self_attn.q_a_layernorm.weight', identity)],
                        identity)
                ]
            )
        else:
            mla_layer_weights.extend([
                WeightInfo(W.mla_q_w, [CkptWeightInfo('model.layers.{i}.self_attn.q_proj.weight', identity)],
                       transpose),
                ]
            )

        layer_weights.extend(mla_layer_weights)
        layer_weights.extend(self._get_hf_ffn_layer_weight_info(layer_id))
        return layer_weights

    def _get_hf_ffn_layer_weight_info(self, layer_id: int):
        inter_padding_size = self._layer_inter_padding_size[layer_id] if self._layer_inter_padding_size else self._inter_padding_size

        if layer_id in self.moe_layer_index_:
            return [
                WeightInfo(W.moe_gate, [CkptWeightInfo('model.layers.{i}.mlp.gate.weight', identity)], transpose),
                WeightInfo(W.ffn_w1, [CkptWeightInfo('model.layers.{i}.mlp.shared_experts.gate_proj.weight', identity)], functools.partial(transpose_pad, inter_padding_size=inter_padding_size, dim=0)),
                WeightInfo(W.ffn_w2, [CkptWeightInfo('model.layers.{i}.mlp.shared_experts.down_proj.weight', identity)], functools.partial(transpose_pad, inter_padding_size=inter_padding_size, dim=1)),
                WeightInfo(W.ffn_w3, [CkptWeightInfo('model.layers.{i}.mlp.shared_experts.up_proj.weight', identity)], functools.partial(transpose_pad, inter_padding_size=inter_padding_size, dim=0)),
                WeightInfo(W.moe_w1, [CkptWeightInfo('model.layers.{i}.mlp.experts.' + str(expert_id) + '.gate_proj.weight', identity) \
                                        for expert_id in range(self.expert_num_)], stack_),
                WeightInfo(W.moe_w2, [CkptWeightInfo('model.layers.{i}.mlp.experts.' + str(expert_id) + '.down_proj.weight',
                                                     functools.partial(multipy_identity, scale=self.routed_scaling_factor)) \
                                        for expert_id in range(self.expert_num_)], stack_),
                WeightInfo(W.moe_w3, [CkptWeightInfo('model.layers.{i}.mlp.experts.' + str(expert_id) + '.up_proj.weight', identity) \
                                        for expert_id in range(self.expert_num_)], stack_),
            ]
        else:
            return [
                WeightInfo(W.ffn_w1, [CkptWeightInfo('model.layers.{i}.mlp.gate_proj.weight', identity)],
                           functools.partial(transpose_pad, inter_padding_size=inter_padding_size, dim=0)),
                WeightInfo(W.ffn_w2, [CkptWeightInfo('model.layers.{i}.mlp.down_proj.weight', identity)],
                           functools.partial(transpose_pad, inter_padding_size=inter_padding_size, dim=1)),
                WeightInfo(W.ffn_w3, [CkptWeightInfo('model.layers.{i}.mlp.up_proj.weight', identity)],
                           functools.partial(transpose_pad, inter_padding_size=inter_padding_size, dim=0)),
            ]

    def _get_weight_info(self):
        layer_weights: List[List[WeightInfo]] = []
        weights = [
            WeightInfo(W.embedding, [CkptWeightInfo('model.embed_tokens.weight', identity)], identity),
            WeightInfo(W.final_ln_gamma, [CkptWeightInfo('model.norm.weight', identity)], identity),
            WeightInfo(W.final_ln_beta, [], functools.partial(zeros, shape=[self._hidden_size])),
            WeightInfo(W.lm_head, [CkptWeightInfo('lm_head.weight', identity)], identity),
        ]
        for layer in range(self._num_layers):
            layer_weights.append(self._get_hf_layer_weight_info(layer))
        return ModelWeightInfo(layer_weights=layer_weights, weights=weights, tp_strategy=W.gpt_style_tp_strategy)


class DeepSeekV2(GPT):
    @classmethod
    def _create_config(cls, ckpt_path: str):
        config = GptInitModelParameters(
            head_num=0,
            head_num_kv=0,
            size_per_head=0,
            layer_num=0,
            inter_size=0,
            vocab_size=102400,
            max_seq_len=8192,
            norm_type='rmsnorm',
            has_post_decoder_layernorm=True)
        config.activation_type = "gated-silu"
        DeepSeekV2._from_hf(config, ckpt_path)
        return config

    @staticmethod
    def _from_hf(config: GptInitModelParameters, ckpt_path: str):
        config_path = os.path.join(ckpt_path, "config.json")
        if not os.path.exists(config_path):
            return
        with open(config_path) as reader:
            content = reader.read()
            config_json = json.loads(content)
            config.inter_size = config_json["intermediate_size"]
            config.head_num = config_json["num_attention_heads"]
            config.head_num_kv = config_json.get("num_key_value_heads", config.head_num)
            config.layer_num = config_json["num_hidden_layers"]
            config.rotary_embedding_base = config_json.get("rope_theta", config.rotary_embedding_base)
            config.vocab_size = config_json["vocab_size"]
            config.layernorm_eps = config_json.get("rms_norm_eps", 1e-06)
            config.tie_word_embeddings = config_json.get('tie_word_embeddings', False)
            config.hidden_size = config_json["hidden_size"]

            # MLA config
            config.use_mla = True
            config.q_lora_rank = config_json['q_lora_rank']
            config.kv_lora_rank = config_json['kv_lora_rank']
            config.nope_head_dim = config_json['qk_nope_head_dim']
            config.rope_head_dim = config_json['qk_rope_head_dim']
            config.v_head_dim = config_json['v_head_dim']
            config.size_per_head = config.nope_head_dim + config.rope_head_dim
            config.rotary_embedding_dim = config.rope_head_dim

            # yarn rotary config
            config.rotary_embedding_style = 5
            rope_scaling = config_json.get('rope_scaling')
            config.rotary_embedding_scale = rope_scaling['factor']
            config.rotary_factor1 = float(rope_scaling.get('beta_slow', 1))
            config.rotary_factor2 = float(rope_scaling.get('beta_fast', 32))
            config.org_embedding_max_pos = rope_scaling['original_max_position_embeddings']

            scaling_factor = rope_scaling['factor']
            mscale = rope_scaling['mscale']
            mscale_all_dim = rope_scaling['mscale_all_dim']
            config.rotary_embedding_mscale = yarn_get_mscale(scaling_factor, mscale) / yarn_get_mscale(scaling_factor, mscale_all_dim)
            config.rotary_embedding_offset = config.nope_head_dim

            # softmax scale config
            softmax_mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
            config.softmax_extra_scale = softmax_mscale * softmax_mscale

            # MOE config
            config.routed_scaling_factor = config_json['routed_scaling_factor']
            config.moe_k = config_json['num_experts_per_tok']
            config.expert_num = config_json['n_routed_experts']
            config.moe_inter_padding_size=config_json['moe_intermediate_size']

            n_shared_experts = config_json['n_shared_experts']
            config.inter_size = n_shared_experts * config.moe_inter_padding_size

            config.layernorm_eps = config_json.get("rms_norm_eps", 1e-06)
            config.has_moe_norm = False
            config.moe_style = 2 # shared + expert

            moe_step = config_json['moe_layer_freq']
            first_k_dense_replace = config_json['first_k_dense_replace']
            config.moe_layer_index = [i for i in range(config.layer_num) if i >= first_k_dense_replace and i % moe_step == 0]

            ffn_inter_size = config_json.get('intermediate_size', config.inter_size)
            layer_inter_size = []
            for i in range(config.layer_num):
                if i in config.moe_layer_index:
                    layer_inter_size.append(config.inter_size)
                else:
                    layer_inter_size.append(ffn_inter_size)
            config.layer_inter_size = layer_inter_size

    @staticmethod
    def get_weight_cls():
        return DeepSeekV2Weight

register_model('deepseek2', DeepSeekV2, ["DeepseekV2ForCausalLM"])
