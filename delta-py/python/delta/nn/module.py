from abc import ABC, abstractmethod

from delta import Tensor, nn


class Module(ABC):

    def __init__(self):
        self._modules: dict[str, "Module"] = {}
        self._parameters: dict[str, "nn.Parameter"] = {}

    @abstractmethod
    def forward(self, *args, **kwargs) -> ...: ...

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def __setattr__(self, name, value):
        if isinstance(value, Module):
            self._modules[name] = value
        elif isinstance(value, nn.Parameter):
            self._parameters[name] = value
        super().__setattr__(name, value)

    def parameters(self) -> list[Tensor]:
        params = list(self._parameters.values())
        for mod in self._modules.values():
            params.extend(mod.parameters())
        return params
