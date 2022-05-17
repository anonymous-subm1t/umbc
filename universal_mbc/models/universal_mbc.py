import torch
from torch import nn

from universal_mbc.models.base import HashableModule
from universal_mbc.models.layers.mbc import Embedder, ResFF, SlotSetEncoder

T = torch.Tensor


class UniversalMBC(HashableModule):
    def __init__(
        self,
        set_encoder: HashableModule,
        K: int,
        h: int,
        d: int,
        d_hat: int,
        slot_type: str = "random",
        ln_slots: bool = False,
        ln_after: bool = False,
        heads: int = 4,
        fixed: bool = False,
        slot_drop: float = 0.0,
        attn_act: str = "sigmoid",
        slot_residual: bool = False,
        embedder: bool = False,
        n_parallel: int = 1
    ):
        super().__init__()
        self.set_encoder = set_encoder

        self.name = f"universal-{set_encoder.name}"
        self.ln_after = ln_after
        self.n_parallel = n_parallel
        self.has_embedder = embedder
        self.set_hashable_attrs(["ln_after", "n_parallel", "has_embedder"])

        self.embedder: nn.Module = nn.Identity()
        if embedder:
            self.embedder = Embedder(d, d_hat)
            h, d = d_hat, d_hat

        self.sse = nn.ModuleList([
            SlotSetEncoder(
                K=K, h=h, d=d, d_hat=d_hat, slot_type=slot_type,
                ln_slots=ln_slots, heads=heads, fixed=fixed, slot_drop=slot_drop,
                attn_act=attn_act, slot_residual=slot_residual
            )
            for i in range(n_parallel)
        ])

        # temporary storage for the minibatch processing forward function
        self.temp = [torch.Tensor() for _ in range(n_parallel)]
        self.temp_c = [torch.Tensor() for _ in range(n_parallel)]
        self.temp_logit_max = [torch.Tensor() for _ in range(n_parallel)]

        # these will all be the identity conditional on PMA usage (the only case where we use ln is when ln_after and not pma)
        self.ff = ResFF(d_hat)
        self.norm_layer1 = nn.LayerNorm(d_hat) if ln_after else nn.Identity()
        self.norm_layer2 = nn.LayerNorm(d_hat) if ln_after else nn.Identity()

    def set_drop_rate(self, p: float) -> None:
        for i, _ in enumerate(self.sse):
            self.sse[i].slot_drop = p

    def mc(self, x: T, samples: int) -> T:
        if self.sse[0].slot_drop <= 0.0:
            raise ValueError(f"cannot call mc when slot drop is off: ({self.sse.slot_drop})")

        x = self.embedder(x)
        out = [self.norm_layer2(self.ff(self.norm_layer1(self.parallel_sse(x)))) for _ in range(samples)]
        out = [self.set_encoder(v) for v in out]
        return torch.stack(out)

    def forward(self, x: T, S: T = None) -> T:
        x = self.embedder(x)
        x = self.parallel_sse(x, S=S)

        x = self.norm_layer1(x)
        x = self.ff(x)
        x = self.norm_layer2(x)

        x = self.set_encoder(x)
        return x

    def parallel_sse(self, x: T, S: T = None) -> T:
        return torch.cat([lyr(x, S=S) for lyr in self.sse], dim=1)

    def get_final_embedding(self) -> T:
        out = []
        for i, _ in enumerate(self.sse):
            view_heads = (self.temp[i].size(0), self.sse[i].slots.K, self.sse[i].heads, -1)
            view_std = (self.temp[i].size(0), self.sse[i].slots.K, -1)

            x = self.temp[i].view(*view_heads) / self.temp_c[i].view(*view_heads)  # type: ignore
            x = x.view(*view_std)
            out.append(x)

        return torch.stack(out)

    def process_minibatch(self, x: T, S: T = None) -> T:
        x = self.embedder(x)

        parallel_s, parallel_x, parallel_c = [], [], []
        for lyr in self.sse:
            _s, _x, _c = lyr.process_batch(x, S=S)
            parallel_s.append(_s)
            parallel_x.append(_x)
            parallel_c.append(_c)

        # store the normalization constant and the unnormalized (QK)V for updating
        out_x = []
        for i, (s, c, x) in enumerate(zip(parallel_s, parallel_c, parallel_x)):
            view_heads = (x.size(0), self.sse[i].slots.K, self.sse[i].heads, -1)
            view_std = (x.size(0), self.sse[i].slots.K, -1)

            self.temp_c[i] = c if (self.temp_c[i].numel() == 0) else self.temp_c[i] + c
            self.temp[i] = x if (self.temp[i].numel() == 0) else self.temp[i] + x

            x = self.temp[i].view(*view_heads) / self.temp_c[i].view(*view_heads)  # type: ignore
            x = x.view(*view_std)
            if self.sse[i].slot_residual:
                x = x + s

            out_x.append(x)

        x = torch.cat(out_x, dim=1)
        x = self.norm_layer1(x)
        x = self.ff(x)
        x = self.norm_layer2(x)

        x = self.set_encoder(x)
        return x

    def reset(self) -> None:
        self.temp = [torch.Tensor().to(self.temp[i].device) for i in range(self.n_parallel)]
        self.temp_c = [torch.Tensor().to(self.temp[i].device) for i in range(self.n_parallel)]
        self.temp_logit_max = [torch.Tensor().to(self.temp[i].device) for i in range(self.n_parallel)]
