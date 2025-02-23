import torch
from abc import abstractmethod
from typing import Any, Optional, Dict, List, Callable, Union

from maga_transformer.utils.model_weight import ModelWeights


class FTWeightsBase:
    def __init__(self):
        self.weights: Union[Dict[str, torch.Tensor], List[torch.Tensor]] = []

    @abstractmethod
    def load(self, *args: List[Any], **kwargs: Any) -> bool:
        raise NotImplementedError

    @property
    def dtype(self):
        if isinstance(self.weights, dict):
            return list(self.weights.values())[0].dtype
        return self.weights[0].dtype

    @property
    def device(self):
        if isinstance(self.weights, dict):
            return list(self.weights.values())[0].device
        return self.weights[0].device

    def _map(self, func: Callable[[torch.Tensor], torch.Tensor]):
        if isinstance(self.weights, dict):
            raise Exception("weight based on map not support _map yet!")
        for i in range(len(self.weights)):
            if isinstance(self.weights[i], list):
                for j in range(len(self.weights[i])):
                    self.weights[i][j] = func(self.weights[i][j])
            else:
                self.weights[i] = func(self.weights[i])

    def float(self):
        if self.dtype == torch.float32:
            return
        self._map(lambda x: x.float())

    def half(self):
        if self.dtype == torch.float16:
            return
        self._map(lambda x: x.half())

    def bfloat16(self):
        if self.dtype == torch.bfloat16:
            return
        self._map(lambda x: x.bfloat16())

    def cuda(self, device: Optional[str]=None):
        self._map(lambda x: x.cuda(device))

    def to(self, device: Optional[str]=None):
        self._map(lambda x: x.to(device))


class FTOPBase:
    def __init__(self):
        self.weight: Optional[ModelWeights] = None
        self.ft_op: Optional[Any] = None

    @classmethod
    def from_config(cls, config: Any) -> Any:
        return cls(config)

    @abstractmethod
    def start(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *args: List[Any], **kwargs: Any) -> Any:
        raise NotImplementedError

    @property
    def dtype(self):
        assert self.weight is not None
        return self.weight.dtype

    @property
    def device(self):
        assert self.weight is not None
        return self.weight.device
