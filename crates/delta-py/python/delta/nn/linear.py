from delta import Tensor
from delta.nn import Module, Parameter
import delta


class Linear(Module):

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()

        self.use_bias = bias
        if self.use_bias:
            in_features += 1
        self.weights = Parameter(delta.randn([in_features, out_features]))

    def forward(self, x: Tensor) -> Tensor:
        if self.use_bias:
            ones_shape = list(x.shape[:-1]) + [1]
            x = delta.cat([x, delta.ones(ones_shape)], dim=-1)
        return x @ self.weights.data
