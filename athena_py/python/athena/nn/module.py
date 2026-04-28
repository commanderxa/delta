from abc import ABC, abstractmethod

from athena import Tensor


class Module(ABC):

    @abstractmethod
    def forward(self, *args, **kwargs) -> ...:
        pass

    def parameters(self) -> list[Tensor]:
        pass