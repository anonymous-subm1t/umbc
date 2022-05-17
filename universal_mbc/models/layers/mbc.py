import math
from functools import partial
from typing import Any, List, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F
from universal_mbc.models.base import HashableModule
from universal_mbc.models.layers.linear import BMM  # type: ignore
from universal_mbc.models.layers.linear import FixedLinear, Linear

T = torch.Tensor


class Slots(HashableModule):
    def __init__(self, K: int, h: int, slot_type: str, fixed: bool = False) -> None:
        super().__init__()
        self.name = "Slots"
        self.K = K                      # Number of Slots
        self.h = h                      # Slot size
        self.slot_type = slot_type      # Deterministic or Random
        self.fixed = fixed              # Fixed or learnable slots
        self.set_hashable_attrs(["slot_type", "fixed", "K", "h"])

        if slot_type not in ["random", "deterministic"]:
            raise ValueError("{} not implemented for slots".format(self.slot_type))

        requires_grad = not fixed
        if slot_type == "random":
            # same initialization as "Weight Uncertainty in Neural Networks"
            self.mu = nn.Parameter(torch.zeros(1, self.K, self.h).uniform_(-0.2, 0.2), requires_grad=requires_grad)
            self.sigma = nn.Parameter(torch.zeros(1, self.K, self.h).uniform_(-5.0, -4.0), requires_grad=requires_grad)
            return

        self.S = nn.Parameter(torch.zeros(1, self.K, self.h), requires_grad=requires_grad)
        nn.init.xavier_uniform_(self.S)  # type: ignore
        # self.S = nn.Parameter(random_features(self.K, self.h, orthogonal=True).unsqueeze(0), requires_grad=requires_grad)

    def sample_s(self) -> T:
        return torch.randn_like(self.mu) * F.softplus(self.sigma) + self.mu


class ResFF(HashableModule):
    def __init__(self, d: int, p: float = 0.0, activation: str = "relu"):
        super().__init__()
        self.name = "ResFF"
        self.p = p
        self.activation = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation]

        self.layer = nn.Sequential(
            nn.Linear(d, d),
            self.activation(),
            nn.Dropout(p=p),
        )

        self.set_hashable_attrs(["p"])

    def forward(self, x: T):
        return x + self.layer(x)


class Embedder(HashableModule):
    def __init__(self, in_dim: int, h_dim: int, activation: str = "relu") -> None:
        super().__init__()
        self.activation = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation]
        self.embedder = nn.Sequential(
            nn.Linear(in_dim, h_dim),
            self.activation(),
        )

    def forward(self, x: T) -> T:
        return self.embedder(x)


class SlotSetEncoder(HashableModule):
    def __init__(
        self,
        K: int,
        h: int,
        d: int,
        d_hat: int,
        eps: float = 1e-8,
        slot_type: str = "deterministic",
        ln_slots: bool = False,
        heads: int = 4,
        fixed: bool = False,
        bias: bool = True,
        slot_drop: float = 0.0,
        attn_act: str = "sigmoid",
        slot_residual: bool = False
    ):
        super().__init__()
        self.name = "SlotSetEncoder"
        self.d = d                                              # Input Dimension
        self.d_hat = d_hat                                      # Linear Projection Dimension
        self.eps = eps                                          # Additive epsilon for stability
        self.heads = heads                                      # number of attention heads
        self.bias = bias
        self.ln_slots = ln_slots
        self.slot_drop = slot_drop
        self.attn_act = attn_act
        self.slot_residual = slot_residual

        self.q: HashableModule
        self.v: HashableModule
        self.k: HashableModule

        self.slots: Slots

        if d_hat % heads != 0:
            raise ValueError(f"for multihead attention, {d_hat=} must be evenly divisible by {heads=}")

        self.sqrt_d_hat = 1.0 / math.sqrt(d_hat // heads)                # Normalization Term
        self.slots = Slots(K=K, h=h, slot_type=slot_type, fixed=fixed)

        lin = {
            False: partial(Linear, in_features=d, out_features=d_hat, bias=bias),
            True: partial(FixedLinear, in_features=d, out_features=d_hat)
        }

        self.q = lin[fixed]()  # type: ignore
        self.v = lin[fixed]()  # type: ignore
        self.k = lin[fixed]()  # type: ignore

        self.norm_slots = nn.LayerNorm(normalized_shape=h) if ln_slots else nn.Identity()
        self.bmm_attn = BMM()
        self.bmm_upda = BMM()
        self.set_hashable_attrs(["heads", "d", "d_hat", "ln_slots", "slot_drop", "attn_act", "slot_residual"])

    def sample_s(self) -> T:
        return self.slots.sample_s() if (self.slots.slot_type == "random") else self.slots.S

    def get_attn_v(self, X: T, S: T = None) -> Tuple[T, T, T]:
        B = X.size(0)
        head_dim = self.d_hat // self.heads

        # Sample Slots Based on B
        if S is None:   # S \in R^{B x K xh}
            S = self.sample_s()

            if self.slot_drop > 0.0:
                idx = torch.rand(self.slots.K) > self.slot_drop
                if idx.sum() == 0:  # we need to ensure that at least one slot is not dropped
                    lucky_one = torch.randperm(self.slots.K)[0]
                    idx[lucky_one] = True

                S = S[:, idx]

        S = self.norm_slots(S).repeat(B, 1, 1)
        assert S is not None

        # Linear Projections k \in R^{B x N x d_hat}, v \in R^{B x N x d_hat}, q \in R^{B x K x d_hat}
        Q, V, K = self.q(X), self.v(X), self.k(S)
        Q = torch.cat(Q.split(head_dim, 2), 0)
        K = torch.cat(K.split(head_dim, 2), 0)
        V = torch.cat(V.split(head_dim, 2), 0)
        return K, self.sqrt_d_hat * self.bmm_attn(K, Q.transpose(1, 2)), V  # M \in R^{B x K x N}

    def get_attn_act(self, W: T, batch_process: bool = False) -> Tuple[T, T]:
        if self.attn_act == "softmax" and batch_process:
            W = torch.exp(W)
        elif self.attn_act == "softmax" and not batch_process:
            logit_max = W.amax(dim=-1, keepdim=True)
            W = torch.exp(W - logit_max)
        elif self.attn_act == "sigmoid":
            W = torch.sigmoid(W)
        elif self.attn_act == "slot-sigmoid":
            W = torch.sigmoid(W)
            W = W / (W.sum(dim=-2, keepdim=True) + self.eps)
        elif self.attn_act == "slot-softmax":
            W = W.softmax(dim=-2)
        elif self.attn_act == "slot-exp":
            W = torch.exp(W - W.amax(dim=-2, keepdim=True))
        else:
            raise NotImplementedError(f"attention activation: {self.attn_act} not implemented")

        C = W.sum(dim=-1, keepdim=True) + self.eps
        if not batch_process:
            return W / C, C  # normalize over N

        # softmax numerical stability requires that we subtract the max from the logit values.
        # so the caller of this function will have to do an extra processing step with this information.
        # if self.attn_act == "softmax":
        #     return W, torch.stack((C, logit_max))
        return W, C

    def forward(self, X: T, S: T = None) -> T:
        B = X.size(0)
        S, W, V = self.get_attn_v(X, S=S)
        A, _ = self.get_attn_act(W)

        S_hat = self.bmm_upda(A, V)     # S_hat \in R^{B x K x D}
        if self.slot_residual:
            S_hat += S

        S_hat = torch.cat(S_hat.split(B, 0), 2)
        return S_hat  # type: ignore

    def process_batch(self, X: T, S: T = None) -> Tuple[T, T, T]:
        B = X.size(0)

        S, W, V = self.get_attn_v(X, S=S)
        W, C = self.get_attn_act(W, batch_process=True)
        S_hat = self.bmm_upda(W, V)     # S_hat \in R^{B x K x D}

        # if self.attn_act == "softmax":
        #     S, S_hat, C, logit_max = map(lambda x: torch.cat(x.split(B, 0), 2), (S, S_hat, C[0], C[1]))
        #     return S, S_hat, torch.stack((C, logit_max))  # type: ignore

        S, S_hat, C = map(lambda x: torch.cat(x.split(B, 0), 2), (S, S_hat, C))
        return S, S_hat, C  # type: ignore


class HierarchicalSSE(HashableModule):
    def __init__(
        self,
        in_dim: int,
        K: List[int],
        h: List[int],
        d: List[int],
        d_hat: List[int],
        heads: List[int],
        slot_drops: List[float],
        g: str = "sum",
        out_dim: int = 5,
        ln_slots: bool = False,
        ln_after: bool = False,
        slot_type: str = "random",
        attn_act: str = "sigmoid",
        slot_residual: bool = True
    ):
        super().__init__()

        self.name = "HierarchicalSSE"
        self.K = K                  # Number of slots in each stage
        self.h = h                  # The dimension of each slot
        self.d = d                  # Input dimension to each stage
        self.g = g                  # Choice of aggregation function g: sum, mean, max, min
        self.d_hat = d_hat          # Projection dimension in each stage
        self.ln_slots = ln_slots    # Use LayerNorm or Not
        self.ln_after = ln_after    # Use LayerNorm or Not

        self.embedder = Embedder(in_dim, d[0]) if in_dim != -1 else nn.Identity()
        sse, ff, norm_ones, norm_twos = [], [], [], []
        for i in range(len(K)):
            sse.append(SlotSetEncoder(
                K=K[i], h=h[i], d=d[i], d_hat=d_hat[i], slot_type=slot_type,
                ln_slots=ln_slots, heads=heads[i], fixed=False, slot_drop=slot_drops[i],
                attn_act=attn_act, slot_residual=slot_residual
            ))

            ff.append(ResFF(d_hat[i]))
            norm_ones.append(nn.LayerNorm(d_hat[i]) if self.ln_after else nn.Identity())
            norm_twos.append(nn.LayerNorm(d_hat[i]) if self.ln_after else nn.Identity())

        def identity(x: T, *args: Any, **kwargs: Any) -> T:
            return x

        self.sse = nn.ModuleList(sse)
        self.ff = nn.ModuleList(ff)
        self.norm_ones = nn.ModuleList(norm_ones)
        self.norm_twos = nn.ModuleList(norm_twos)

        self.poolfuncs: Any = {"mean": torch.mean, "sum": torch.sum, "min": torch.amin, "max": torch.amax, "identity": identity}
        self.set_hashable_attrs(["ln_after", "g"])

        self.lin_out = nn.Linear(d_hat[-1] if g != "cat" else d_hat[-1] * K[-1], out_dim)

    def pool(self, x: T) -> T:
        if self.g == "cat":
            return x.view(x.size(0), 1, x.size(1) * x.size(2))
        return self.poolfuncs[self.g](x, dim=1, keepdim=True)

    def forward(self, x: T, split_size: int = None, S: List[T] = None) -> T:
        x = self.embedder(x)
        if split_size is None:
            for i, (lyr, ff, norm1, norm2) in enumerate(zip(self.sse, self.ff, self.norm_ones, self.norm_twos)):
                x = lyr(x, S=S[i] if S is not None else None)
                x = norm2(ff(norm1(x)))
        else:
            B = x.size(0)
            x = torch.split(x, split_size_or_sections=split_size, dim=1)

            # The final matrix multiplication is (K, N) @ (N, D) which is in the form
            # AA^T which can be expressed as a sum of aa^T. So we need to store the intermediate
            # (K, D) matrix because the output of the first encoder layer will affect latter inputs
            # NOTE:This Does not affect training! It is only for MBC testing of stacked SSE's
            tmp_slots, tmp, tmp_c = torch.Tensor(), torch.Tensor(), torch.Tensor()
            for i, partition in enumerate(x):
                for j, (lyr, ff, norm1, norm2) in enumerate(zip(self.sse, self.ff, self.norm_ones, self.norm_twos)):
                    if i == 0 and j == 0:
                        # if tmp is unset, (i == 0 and j == 0) set it here so we can use it on the next partition
                        slots, partition_unnorm, c = lyr.process_batch(partition, S=S[j] if S is not None else None)
                        partition = (partition_unnorm.view(B, lyr.slots.K, lyr.heads, -1) / c.view(B, lyr.slots.K, lyr.heads, 1)).view(B, lyr.slots.K, -1)
                        tmp_slots, tmp, tmp_c = slots, partition_unnorm, c

                        # unroll to line up the heads, normalize and then divide. continue onto next loop where we can use this MBC encoding
                    elif i > 0 and j == 0:
                        # if this is not the first partition, but this is the first encoder layer,
                        # then we need to sum the partition with the previously stored representation
                        # (N, K, D). For every encoder layer after the first one, the changes should
                        # cascade so we do not have to do anything for j > 0
                        slots, partition_unnorm, c = lyr.process_batch(partition, S=S[j] if S is not None else None)
                        tmp += partition_unnorm
                        tmp_c += c
                        partition = tmp_slots + (tmp.view(B, lyr.slots.K, lyr.heads, -1) / tmp_c.view(B, lyr.slots.K, lyr.heads, 1)).view(B, lyr.slots.K, -1)
                    else:
                        partition = lyr(partition, S=S[j] if S is not None else None)

                    partition = norm2(ff(norm1(partition)))
            x = partition

        # This is for the modelnet models where we dont necessarily have to directly go to 1.
        if x.size(1) > 1:
            x = self.pool(x)
        return self.lin_out(x.squeeze(1))
