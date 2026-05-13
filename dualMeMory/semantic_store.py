import torch
from torch import Tensor
from torch.nn import Module, init
from torch.nn.parameter import Parameter


class SemanticStore(Module):
    """Semantic correlation-matrix memory C_sem of shape (d, d).

    Storage / retrieval convention (matches MeMo's existing CMM_OUT, so that
    blended retrieval with C_epi is well-defined):

        write:    C_sem += outer(key, value)
        read:     predicted_value = key_query @ C_sem

    The dual-trace spec uses two write primitives on top of this:
      - consolidate(token x_i, context c_i): key = c_i, value = x_i
      - teach_association(a, b):              key = a,   value = b
    """

    def __init__(self, d: int, device=None, dtype=None, init_weights: bool = True):
        super().__init__()
        self.d = d
        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = Parameter(
            torch.empty((d, d), **factory_kwargs), requires_grad=False
        )
        if init_weights:
            self.reset_parameters()

    def reset_parameters(self) -> None:
        init.zeros_(self.weight)

    def forward(self, key_query: Tensor) -> Tensor:
        return torch.matmul(key_query, self.weight)

    def _pair_update(self, keys: Tensor, values: Tensor) -> Tensor:
        k = keys.reshape(-1, self.d)
        v = values.reshape(-1, self.d)
        return k.transpose(0, 1) @ v

    def add_pairs(self, keys: Tensor, values: Tensor, scale: float = 1.0) -> None:
        update = self._pair_update(keys, values)
        with torch.no_grad():
            self.weight.add_(update, alpha=scale)

    def sub_pairs(self, keys: Tensor, values: Tensor, scale: float = 1.0) -> None:
        update = self._pair_update(keys, values)
        with torch.no_grad():
            self.weight.sub_(update, alpha=scale)

    def add_association(self, a_vec: Tensor, b_vec: Tensor, scale: float = 1.0) -> None:
        update = torch.outer(a_vec.flatten(), b_vec.flatten())
        with torch.no_grad():
            self.weight.add_(update, alpha=scale)

    def sub_association(self, a_vec: Tensor, b_vec: Tensor, scale: float = 1.0) -> None:
        update = torch.outer(a_vec.flatten(), b_vec.flatten())
        with torch.no_grad():
            self.weight.sub_(update, alpha=scale)

    def association_strength(self, a_vec: Tensor, b_vec: Tensor) -> Tensor:
        return a_vec.flatten() @ self.weight @ b_vec.flatten()
