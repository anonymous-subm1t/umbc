from functools import partial
from typing import List, Type

import torch
import torch.nn as nn

from universal_mbc.models.base import HashableModule
from universal_mbc.models.layers.linear import Linear  # type: ignore
from universal_mbc.models.layers.linear import get_linear_layer
from universal_mbc.models.layers.mbc import HierarchicalSSE  # type: ignore
from universal_mbc.models.layers.set_xformer import ISAB, PMA

T = torch.Tensor


class MBC(HashableModule):
    def __init__(
        self,
        in_dim: int = 3,
        out_dim: int = 40,
        hidden_dim: int = 256,
        num_layers: int = 3,
        extractor: str = 'PermEquiMax',
        activation: Type[nn.Module] = nn.Tanh,
        K: List[int] = [16],
        h: List[int] = [256],
        d: List[int] = [256],
        d_hat: List[int] = [256],
        heads: List[int] = [4],
        slot_drops: List[float] = [0.0],
        g: str = 'max',
        ln_slots: bool = True,
        ln_after: bool = True,
        slot_type: str = 'random',
        slot_residual: bool = True,
        attn_act: str = "slot-softmax",
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.extractor = extractor
        self.name = 'MBC'

        linear = get_linear_layer(extractor)

        features: List[nn.Module] = []
        for i in range(num_layers):
            features.append(linear(in_dim if i == 0 else hidden_dim, hidden_dim))
            features.append(activation())

        self.phi = nn.Sequential(*features)
        self.ro = nn.Sequential(
            HierarchicalSSE(
                -1, K=K, h=h, d=d, d_hat=d_hat, heads=heads, out_dim=d_hat[-1],
                slot_drops=slot_drops, g=g, ln_slots=ln_slots, ln_after=ln_after,
                slot_type=slot_type, attn_act=attn_act, slot_residual=slot_residual
            ),
            nn.Dropout(),
            Linear(hidden_dim, hidden_dim),
            activation(),
            nn.Dropout(),
            Linear(hidden_dim, out_dim),
        )

        self.num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.set_hashable_attrs(["extractor", "hidden_dim", "num_layers"])

    def get_pooled(self, x: T) -> T:
        features = self.phi(x)
        return self.ro[0](features)  # type: ignore

    def forward(self, x: T) -> T:
        features = self.phi(x)
        pred = self.ro(features)
        return pred  # type: ignore


class DeepSets(HashableModule):
    def __init__(
        self,
        hidden_dim: int = 256,
        x_dim: int = 3,
        out_dim: int = 40,
        num_layers: int = 3,
        extractor: str = 'PermEquiMax',
        activation: Type[nn.Module] = nn.Tanh,
        pool: str = "max"
    ):
        super().__init__()
        self.name = 'DeepSets'
        self.hidden_dim = hidden_dim
        self.extractor = extractor
        self.pool = pool
        self.x_dim = x_dim
        self.num_layers = num_layers
        self.out_dim = out_dim

        linear = get_linear_layer(extractor)
        self.pool_func = {"min": torch.amin, "max": torch.amax, "mean": torch.mean, "sum": torch.sum}[pool]

        phi: List[nn.Module] = []
        for i in range(num_layers):
            phi.append(linear(x_dim if i == 0 else hidden_dim, hidden_dim))
            phi.append(activation())

        self.phi = nn.Sequential(*phi)
        self.ro = nn.Sequential(
            nn.Dropout(),
            Linear(self.hidden_dim, self.hidden_dim),
            activation(),
            nn.Dropout(),
            Linear(self.hidden_dim, self.out_dim),
        )

        self.num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.set_hashable_attrs(["extractor", "pool", "num_layers"])

    def get_pooled(self, x: T) -> T:
        phi_output = self.phi(x)
        return self.pool_func(phi_output, dim=1)  # type: ignore

    def forward(self, x: T) -> T:
        phi_output = self.phi(x)
        max_output = self.pool_func(phi_output, dim=1)  # type: ignore
        pred = self.ro(max_output)
        return pred  # type: ignore


class MaxPooler(nn.Module):
    def forward(self, x: T) -> T:
        return torch.amax(x, dim=1, keepdim=True)


class SetXformer(HashableModule):
    def __init__(
        self,
        dim_input=3,
        num_outputs=1,
        dim_output=40,
        num_inds=16,
        hidden_dim=256,
        num_heads=4,
        n_isab=2,
        ln=True,
        pool: str = "max"
    ):
        super().__init__()
        self.name = 'SetTransformer'
        self.dim_input = dim_input
        self.num_outputs = num_outputs
        self.dim_output = dim_output
        self.num_inds = num_inds
        self.hidden_dim = hidden_dim
        self.pool = pool
        self.num_heads = num_heads
        self.ln = ln

        self.set_hashable_attrs(["pool"])

        enc = []
        for i in range(n_isab):
            enc.append(ISAB(hidden_dim if i != 0 else dim_input, hidden_dim, num_heads, num_inds, ln=ln))

        self.enc = nn.Sequential(*enc)

        def get_dec() -> nn.Module:
            if pool == "pma":
                return nn.Sequential(
                    nn.Dropout(),
                    PMA(hidden_dim, num_heads, num_outputs, ln=ln),
                    nn.Dropout(),
                    Linear(hidden_dim, dim_output)
                )
            return nn.Sequential(
                MaxPooler(),
                nn.Dropout(),
                Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(),
                Linear(hidden_dim, dim_output)
            )

        self.dec = get_dec()
        self.num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_pooled(self, x: T) -> T:
        x = self.enc(x)
        if self.pool == "pma":
            return self.dec[1](self.dec[0](x))  # type: ignore
        return self.dec[0](x)  # type: ignore

    def forward(self, X):
        pred = self.dec(self.enc(X)).squeeze()
        return pred


def clip_grad(model: nn.Module, max_norm: int) -> float:
    total_norm = 0.0
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm ** 2

    total_norm = total_norm ** (0.5)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in model.parameters():
            p.grad.data.mul_(clip_coef)

    return total_norm
