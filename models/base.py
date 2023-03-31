from typing import List, Any
from torch import nn, tensor
from abc import abstractmethod

class BaseVAE(nn.Module):

    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    def encode(self, input: tensor) -> List[tensor]:
        raise NotImplementedError

    def decode(self, input: tensor) -> Any:
        raise NotImplementedError

    def sample(self, batch_size:int, **kwargs) -> tensor:
        raise NotImplementedError

    def generate(self, x: tensor, **kwargs) -> tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: tensor) -> tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> tensor:
        pass