import logging
import os
import time
from pathlib import Path
from typing import List, Union

import torch
from torch import nn

from maga_transformer.distribute.worker_info import g_parallel_info
from maga_transformer.models.multimodel.multimodel_common import ImageTransform, ImageEmbeddingInterface

try:
    import tensorrt as trt
except ImportError as e:
    pass

def torch_dtype_from_trt(dtype):
    # TODO(xyz): support quantization such as int8, int4
    if dtype == trt.bfloat16:
        return torch.bfloat16
    elif dtype == trt.float16:
        return torch.float16
    elif dtype == trt.float32:
        return torch.float32
    else:
        raise TypeError(f"unsupported tensorrt data type {dtype}")


def torch_device_from_trt(device):
    if device == trt.TensorLocation.DEVICE:
        return torch.device("cuda")
    elif device == trt.TensorLocation.HOST:
        return torch.device("cpu")
    else:
        return TypeError(f"unsupported tensorrt device type {device}")


def torch_type_to_path(dtype: torch.dtype):
    if dtype == torch.float16:
        return "fp16"
    elif dtype == torch.bfloat16:
        return "bf16"
    elif dtype == torch.float32:
        return "fp32"
    else:
        raise TypeError(f"unknown torch data type {dtype}")

# TODO(xyz): make multimodel trt engine more general, not only handle the image case
class MultiModelTRTEngine(nn.Module, ImageEmbeddingInterface):
    def __init__(
        self,
        model_name: str,
        image_size: int,
        device: Union[str, torch.device],
        dtype: torch.dtype,
    ):
        super(MultiModelTRTEngine, self).__init__()
        self.image_size = image_size
        self.image_transform = ImageTransform(self.image_size)
        self.device = device
        self.dtype = dtype
        self.max_batch_size = 1
        self.cur_batch_size = 1
        self.input_names = ["input"]
        self.output_names = ["output"]
        output_dir = MultiModelTRTEngine.cache_path(model_name, self.dtype)
        self.onnx_file_path = os.path.join(output_dir, "multimodel.onnx")
        self.engine_file_path = os.path.join(output_dir, "multimodel.trt")
        self.engine = None

    @staticmethod
    def trt_engine_cached(model_name: str, dtype: torch.dtype) -> bool:
        return not MultiModelTRTEngine.completion_file_path(model_name, dtype).exists()

    @staticmethod
    def cache_path(model_name: str, dtype: torch.dtype) -> str:
        return os.path.join(
            os.environ.get("TRT_CACHE_PATH", os.path.join(os.getcwd(), "trt_cache")),
            f"{model_name}_{torch_type_to_path(dtype)}",
        )

    @staticmethod
    def completion_file_path(model_name: str, dtype: torch.dtype) -> Path:
        return Path(
            os.path.join(
                MultiModelTRTEngine.cache_path(model_name, dtype), "vit_trt.done"
            )
        )

    def export_onnx(
        self,
        network: torch.nn.Module,
    ):
        logging.info("Start exporting torch to ONNX model")
        image = (
            torch.randn(self.cur_batch_size, 3, self.image_size, self.image_size)
            .to(self.device)
            .to(self.dtype)
        )
        with torch.inference_mode():
            torch.onnx.export(
                network,
                image,
                self.onnx_file_path,
                opset_version=17,
                input_names=self.input_names,
                output_names=self.output_names,
            )
        logging.info("Finish exporting ONNX model")

    def generate_trt_engine(self):
        logging.info("Start generating TRT engine!")

        logger = trt.Logger(trt.Logger.VERBOSE)
        builder = trt.Builder(logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        config = builder.create_builder_config()

        if self.dtype == torch.float32:
            pass
        elif self.dtype == torch.float16:
            config.set_flag(trt.BuilderFlag.FP16)
        elif self.dtype == torch.bfloat16:
            config.set_flag(trt.BuilderFlag.BF16)
        else:
            raise ValueError(f"unsupported torch data type {self.dtype}")

        profile = builder.create_optimization_profile()

        # parse onnx model
        parser = trt.OnnxParser(network, logger)
        with open(self.onnx_file_path, "rb") as model:
            if not parser.parse(model.read(), "/".join(self.onnx_file_path.split("/"))):
                logging.info(f"Failed parsing {self.onnx_file_path}")
                for error in range(parser.num_errors):
                    logging.info(parser.get_error(error))
            logging.info(f"Succeeded parsing {self.onnx_file_path}")

        nBS = -1
        nMinBS = 1
        nOptBS = max(1, int(self.max_batch_size / 2))
        nMaxBS = self.max_batch_size
        inputT = network.get_input(0)
        inputT.shape = [nBS, 3, self.image_size, self.image_size]
        profile.set_shape(
            inputT.name,
            [nMinBS, 3, self.image_size, self.image_size],
            [nOptBS, 3, self.image_size, self.image_size],
            [nMaxBS, 3, self.image_size, self.image_size],
        )

        config.add_optimization_profile(profile)

        t0 = time.time()
        engineString = builder.build_serialized_network(network, config)
        t1 = time.time()
        if engineString is None:
            logging.info(f"Failed building {self.engine_file_path}")
        else:
            logging.info(f"Succeeded building {self.engine_file_path} in {t1 - t0} s")
        with open(self.engine_file_path, "wb") as f:
            f.write(engineString)

    def load_trt_engine(self):
        logging.info("Start loading TRT engine!")
        G_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(self.engine_file_path, "rb") as f, trt.Runtime(G_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        logging.info("Start initializing TRT engine!")
        if self.engine is not None:
            self.bindings = [None] * (len(self.input_names) + len(self.output_names))
            self.context = self.engine.create_execution_context()

            # fix input shape
            input_shape = self.engine.get_tensor_shape(self.input_names[0])
            input_shape[0] = self.cur_batch_size
            self.context.set_input_shape(self.input_names[0], tuple(input_shape))

            self.output_shape = tuple(
                self.engine.get_tensor_shape(self.output_names[0])
            )
            self.output_dtype = torch_dtype_from_trt(
                self.engine.get_tensor_dtype(self.output_names[0])
            )
            self.output_device = torch_device_from_trt(
                self.engine.get_tensor_location(self.output_names[0])
            )
        else:
            raise ValueError(f"Failed loading {self.engine_file_path}")


    def forward(self, *inputs):
        input = inputs[0]
        batch_size = input.shape[0]

        # update input shape
        if batch_size != self.cur_batch_size:
            input_shape = self.engine.get_tensor_shape(self.input_names[0])
            input_shape[0] = self.cur_batch_size
            self.context.set_input_shape(self.input_names[0], tuple(input_shape))
            self.output_shape = tuple(
                self.engine.get_tensor_shape(self.output_names[0])
            )
            self.output_shape[0] = self.cur_batch_size

        output = torch.empty(
            size=self.output_shape, dtype=self.output_dtype, device=self.output_device
        )

        # ensure the input tensor passed into trt engine is continous in memory,
        # if not, change the input tensor to be continous
        self.bindings[0] = input.data_ptr()
        self.bindings[1] = output.data_ptr()

        # execute the engine synchronously
        self.context.execute_v2(self.bindings)

        return output

    def encode(self, image_paths: List[str], device: Union[str, torch.device]):
        images = self.image_transform.encode(image_paths, device, self.dtype)
        return self(images)

    def image_embedding(
        self, images: List[str], device: Union[str, torch.device]
    ) -> torch.Tensor:
        # TRT engine doesn't support TP, here we only transform image in rank 0,
        # later input embedding result will be broadcast from rank 0 in async_input_word_embedding
        if g_parallel_info.tp_rank == 0:
            if len(images) != 0:
                images = self.encode(images, device)
            return images.to(device=device)
        else:
            return torch.zeros((len(images), self.output_shape[1:]), self.dtype).to(device=device)
