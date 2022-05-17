import math

import torch
import torch.nn as nn
from universal_mbc.models.base import HashableModule
from universal_mbc.models.layers.mbc import ResFF

T = torch.Tensor

class MAB(HashableModule):
    def __init__(self, dim_Q: int, dim_K: int, dim_V: int, num_heads: int, ln: bool = False):
        super().__init__()
        self.dim_Q = dim_Q
        self.dim_K = dim_K
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.ln = ln

        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        # self.fc_o = nn.Linear(dim_V, dim_V)
        self.fc_o = ResFF(dim_V)

        self.set_hashable_attrs(["dim_Q", "dim_K", "dim_V", "num_heads", "ln"])

    def forward(self, Q: T, K: T) -> T:
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V // self.num_heads), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        # O = O + F.relu(self.fc_o(O))
        O = self.fc_o(O)
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O

    def extra_repr(self) -> str:
        return f"Q={self.dim_Q}, K={self.dim_K}, V={self.dim_V}, num_heads={self.num_heads}, ln={self.ln}"


class SAB(HashableModule):
    def __init__(self, dim_in: int, dim_out: int, num_heads: int, ln: bool = False):
        super().__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)
        self.ln = ln

        self.set_hashable_attrs(["ln"])

    def forward(self, X: T) -> T:
        return self.mab(X, X)


class ISAB(HashableModule):
    def __init__(self, dim_in: int, dim_out: int, num_heads: int, num_inds: int, ln: bool = False):
        super().__init__()
        self.num_inds = num_inds

        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

        self.set_hashable_attrs(["num_inds"])

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        H = self.mab1(X, H)
        return H


class PMA(HashableModule):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_seeds: int,
        ln: bool = False,
        dim_Q: int = None,
        dim_K: int = None,
        dim_V: int = None
    ):
        super().__init__()
        self.num_seeds = num_seeds

        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim_Q or dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim_Q or dim, dim_K or dim, dim_V or dim, num_heads, ln=ln)

        self.set_hashable_attrs(["num_seeds"])

    def forward(self, X: T) -> T:
        X = self.mab(self.S.repeat(X.size(0), 1, 1), X)
        return X
