from typing import Any, List, Type

import torch
from torch import nn

from universal_mbc.models.base import HashableModule
from universal_mbc.models.layers.linear import get_linear_layer  # type: ignore
from universal_mbc.models.layers.mbc import ResFF  # type: ignore
from universal_mbc.models.layers.mbc import SlotSetEncoder
from universal_mbc.models.layers.set_xformer import ISAB, PMA, SAB

T = torch.Tensor


class DeepSets(HashableModule):
    def __init__(
        self, in_dim: int,
        h_dim: int,
        out_dim: int,
        K: int,
        x_depth: int,
        d_depth: int,
        agg: str = "mean",
        linear_type: str = "Linear",  # PermEqui... layers failed to learn on this experiment for some reason
        activation: Type[nn.Module] = nn.ReLU
    ) -> None:
        super().__init__()

        self.name = "DeepSets"
        self.K = K
        self.agg_str = agg
        self.linear_type = linear_type
        self.agg_func = {"max": torch.amax, "min": torch.amin, "mean": torch.mean, "sum": torch.sum}[agg]
        self.set_hashable_attrs(["agg_str", "linear_type"])

        linear = get_linear_layer(linear_type)

        x: List[Any] = []
        for i in range(x_depth):
            x.extend([linear(in_dim if i == 0 else h_dim, h_dim), activation()])

        e = linear(h_dim, h_dim)

        d: List[Any] = []
        for i in range(d_depth):
            d.extend([nn.Dropout(), nn.Linear(h_dim, h_dim), activation()])
        d.append(nn.Linear(h_dim, K * out_dim))

        self.extractor = nn.Sequential(*x)
        self.encoder = e
        self.decoder = nn.Sequential(*d)

    def get_pooled(self, x: T) -> T:
        x = self.extractor(x)
        return self.agg_func(self.encoder(x), dim=1)  # type: ignore

    def forward(self, x: T) -> T:
        x = self.extractor(x)
        x = self.agg_func(self.encoder(x), dim=1)  # type: ignore
        return self.decoder(x).view(x.size(0), self.K, -1)  # type: ignore


class MBC(HashableModule):
    def __init__(
        self,
        in_dim: int,
        h_dim: int,
        out_dim: int,
        K: int,
        x_depth: int,
        d_depth: int,
        heads: int = 4,
        ln_slots: bool = False,
        ln_after: bool = True,
        slot_type: str = "deterministic",
        slot_drop: float = 0.0,
        attn_act: str = "sigmoid",
        slot_residual: bool = False,
        activation: Type[nn.Module] = nn.ReLU,
        sab_decoder: bool = False
    ) -> None:
        super().__init__()

        self.name = "MBC"
        self.ln_after = ln_after
        self.activation = activation

        self.set_hashable_attrs(["ln_after", "activation"])

        x: List[Any] = []
        for i in range(x_depth):
            x.extend([nn.Linear(in_dim if i == 0 else h_dim, h_dim), activation()])

        e: List[Any] = [
            SlotSetEncoder(
                K, h_dim, h_dim, h_dim, slot_type=slot_type,
                ln_slots=ln_slots, heads=heads, fixed=False, slot_drop=0.0,
                attn_act=attn_act, slot_residual=slot_residual
            )
        ]

        self.ff = ResFF(h_dim)
        self.norm_layer1 = nn.LayerNorm(h_dim) if ln_after else nn.Identity()
        self.norm_layer2 = nn.LayerNorm(h_dim) if ln_after else nn.Identity()

        d: List[Any] = []
        for i in range(d_depth):
            if sab_decoder:
                d.append(SAB(dim_in=h_dim, dim_out=h_dim, num_heads=heads, ln=True))
                continue

            d.extend([nn.Linear(h_dim, h_dim), activation()])
        d.append(nn.Linear(h_dim, out_dim))

        self.extractor = nn.Sequential(*x)
        self.encoder = nn.Sequential(*e)
        self.decoder = nn.Sequential(*d)

    def get_pooled(self, x: T) -> T:
        x = self.extractor(x)
        return self.encoder(x)

    def forward(self, x: T) -> T:
        x = self.extractor(x)
        x = self.encoder(x)

        x = self.norm_layer1(x)
        x = self.ff(x)
        x = self.norm_layer2(x)

        # (B, K, D)
        return self.decoder(x)  # type: ignore


class SetXformer(HashableModule):
    def __init__(
        self,
        in_dim: int,
        h_dim: int,
        out_dim: int,
        K: int,
        x_depth: int,
        d_depth: int,
        num_heads: int = 4,
        num_inds: int = 16,
        ln: bool = True,
        isab_enc: bool = True,
    ) -> None:
        super().__init__()
        # the original repo did not use layernorm in their experiments, but we could not reproduce their results without
        # it so we included it here

        self.name = "SetXformer"
        self.K = K
        self.num_heads = num_heads
        self.temp = torch.Tensor()
        self.temp_count = 0

        x: List[Any] = []
        for i in range(x_depth):
            if isab_enc:
                x.append(ISAB(dim_in=in_dim if i == 0 else h_dim, dim_out=h_dim, num_heads=num_heads, num_inds=num_inds, ln=ln))
                continue

            x.extend([nn.Linear(in_dim if i == 0 else h_dim, h_dim), nn.ReLU()])

        e = nn.Sequential(PMA(dim=h_dim, num_heads=num_heads, num_seeds=K, ln=ln))

        d: List[Any] = []
        for i in range(d_depth):
            d.append(SAB(dim_in=h_dim, dim_out=h_dim, num_heads=num_heads, ln=ln))
        d.append(nn.Linear(h_dim, out_dim))

        self.extractor = nn.Sequential(*x)
        self.encoder = e
        self.decoder = nn.Sequential(*d)

    def process_minibatch(self, x: T) -> T:
        x = self.extractor(x)
        x = self.encoder(x)  # type: ignore

        self.temp = x if (self.temp.numel() == 0) else self.temp + x
        self.temp_count += 1

        return self.decoder(self.temp / self.temp_count)

    def get_final_embedding(self) -> T:
        return self.temp / self.temp_count

    def reset(self) -> None:
        self.temp = torch.Tensor().to(self.temp.device)
        self.temp_count = 0

    def get_pooled(self, x: T) -> T:
        x = self.extractor(x)
        return self.encoder(x)  # type: ignore

    def forward(self, x: T) -> T:
        x = self.extractor(x)
        x = self.encoder(x)  # type: ignore
        x = self.decoder(x)  # type: ignore
        return x
