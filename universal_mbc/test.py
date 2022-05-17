import os
import time
import unittest
from argparse import Namespace
from functools import partial
from typing import Any

import numpy as np
import pandas as pd  # type: ignore
import torch
from torch import nn
from utils import get_module_root, md5

from universal_mbc.models.classification import MBC, DeepSets
from universal_mbc.models.classification import SetXformer as SetXformerClass
from universal_mbc.models.diem.mog_models import EmbedderMoG
from universal_mbc.models.layers.mbc import HierarchicalSSE, SlotSetEncoder
from universal_mbc.models.mvn import SetXformer
from universal_mbc.models.universal_mbc import UniversalMBC

T = torch.Tensor


def check_close(A: T, B: T) -> bool:
    result = torch.all(torch.isclose(A, B, rtol=0.0, atol=1e-4))
    if not result:
        print(f"failed (err: {torch.abs(A - B).amax()})")
    return result  # type: ignore


class SSETester(SlotSetEncoder):
    def check_minibatch_consistency(self, X: T, split_size: int) -> bool:
        # Sample Slots for Current S, encode the full set
        B, _, _, device = *X.size(), X.device
        S = self.sample_s()
        S_hat_X = self.forward(X=X, S=S)

        # Split X each with split_size elements, Encode splits.
        X = torch.split(X, split_size_or_sections=split_size, dim=1)
        S_hat_split, C = torch.zeros(B, self.slots.K, self.d_hat, device=device), torch.zeros(B, self.slots.K, self.heads, device=device)

        view_heads = (B, self.slots.K, self.heads, -1)
        view_std = (B, self.slots.K, -1)

        for split_i in X:
            S_out, S_hat_split_i, C_i = self.process_batch(X=split_i, S=S)

            S_hat_split += S_hat_split_i
            C += C_i

        S_hat_split = (S_hat_split.view(*view_heads) / C.view(*view_heads)).view(*view_std)

        if self.slot_residual:
            S_hat_split += S_out

        return check_close(S_hat_X, S_hat_split)

    def check_input_invariance(self, X: T) -> bool:
        # Sample Slots for Current S, encode full set
        B, s, _ = X.size()
        S = self.sample_s()
        S_hat = self.forward(X=X, S=S)

        # Random permutations on X
        permutations = torch.randperm(s)
        X = X[:, permutations, :]

        S_hat_perm = self.forward(X=X, S=S)
        return check_close(S_hat, S_hat_perm)

    def check_slot_equivariance(self, X: T) -> bool:
        # Encode full set
        S = self.sample_s()
        S_hat = self.forward(X=X, S=S)

        # Random permutations on S
        permutations = torch.randperm(self.slots.K)
        S = S[:, permutations, :]
        S_hat_perm = self.forward(X=X, S=S)

        # Apply sampe permutation on S_hat
        S_hat = S_hat[:, permutations, :]
        return check_close(S_hat, S_hat_perm)


class TestDiem(unittest.TestCase):
    def test_smoketest_diem_mog(self) -> None:
        args = Namespace(out_type="select_best2", D=2, K=4, num_proto=4, num_ems=3, dim_feat=128, num_heads=5, tau=10.0)
        distr_emb_args = {
            "dh": 128, "dout": 64, "num_eps": 5, "layers1": [128],
            "nonlin1": "relu", "layers2": [128], "nonlin2": "relu",
        }

        net = EmbedderMoG(
            args.D, args.K, out_type=args.out_type, num_proto=args.num_proto, num_ems=args.num_ems,
            dim_feat=args.dim_feat, num_heads=args.num_heads, tau=args.tau, distr_emb_args=distr_emb_args
        )
        net = net.cuda()
        x = torch.randn(32, 100, 2).cuda()

        out = net(x)
        self.assertTrue(all([u == v for (u, v) in zip(out.size(), [32, 4, 5])]))


class TestUniversalMBC(unittest.TestCase):
    @unittest.skip("skipping because this was only used to generate data for one figure")
    def test_embedding_vec_ll(self) -> None:
        dim = 128
        encoder = partial(SetXformer, in_dim=dim, h_dim=dim, out_dim=dim, K=1, x_depth=2, d_depth=2, num_heads=1)
        umbc_encoder = partial(SetXformer, in_dim=dim, h_dim=dim, out_dim=dim, K=1, x_depth=2, d_depth=2, num_heads=1)

        model = encoder().cuda()
        umbc_model = UniversalMBC(umbc_encoder(), K=1, h=dim, d=dim, d_hat=dim, heads=1, fixed=False, slot_type="deterministic", ln_after=True, n_parallel=1).cuda()  # type: ignore
        umbc_model.eval()  # make sure dropout is off
        model.eval()

        embeddings = []
        data = {"chunk_size": [], "universal": []}  # type: ignore
        with torch.no_grad():
            for i in range(100):  # 100 random samples of tensor
                x = torch.randn(1, 1024, dim).cuda()
                for chunks in [2, 4, 8, 16, 32, 64]:  # 6 chunk sizes
                    x_perm = x[torch.randperm(x.size(0))]
                    x_chunked = x_perm.chunk(chunks, dim=1)
                    for mdl, universal in zip([model, umbc_model], [False, True]):
                        mdl.reset()  # type: ignore
                        for chnk in x_chunked:
                            mdl.process_minibatch(chnk)  # type: ignore

                        out = mdl.get_final_embedding().cpu().view(-1).numpy()  # type: ignore
                        embeddings.append(out)
                        data["chunk_size"].append(chunks)
                        data["universal"].append(universal)
                        print(".", end="", flush=True)

        df = pd.DataFrame(data)
        root = get_module_root()
        df.to_csv(os.path.join(root, "universal_mbc", "plots", "st-umbcst-embeddings.csv"))
        arr = np.stack(embeddings)
        np.save(os.path.join(root, "universal_mbc", "plots", "st-umbcst-embeddings.npy"), arr)

    @unittest.skip("skipping because this was only used to generate data for one figure")
    def test_universal_dropout_time_savings(self) -> None:
        encoder = partial(SetXformerClass, dim_input=128, hidden_dim=128)
        with torch.no_grad():
            umbc = UniversalMBC(
                encoder(), K=2**12, h=3, d=3,  # type: ignore
                d_hat=128, heads=4, fixed=False,
                slot_type="deterministic", ln_after=True,
                n_parallel=1, attn_act="softmax"
            )
            umbc.eval()  # make sure dropout is off

            umbc = umbc.cuda()
            x = torch.randn(32, 200, 3).cuda()

            out = []
            for p in range(1, 100):
                print(f"testing: {p}")
                umbc.set_drop_rate(1 / p)
                times = []
                for _ in range(250):
                    start = time.time()
                    _ = umbc(x)
                    times.append(time.time() - start)
                out.append(times)

            arr_out = np.array(out)

            root = get_module_root()
            np.save(os.path.join(root, "universal_mbc", "plots", "drop-times.npy"), arr_out)

    def test_universal_hash_collisions(self) -> None:
        hashes = []
        for (set_encoder, name) in zip([nn.Linear(1, 1), nn.Linear(1, 2)], ["one", "two"]):
            for ln_after in [True, False]:
                for n_parallel in [1, 2]:
                    set_encoder.name = name  # type: ignore
                    model = UniversalMBC(set_encoder, K=1, h=4, d=1, d_hat=4, ln_after=ln_after, n_parallel=n_parallel)  # type: ignore
                    hashes.append(md5(str(model)))

        self.assertTrue(len(hashes) == len(list(set(hashes))))
        self.assertTrue(all([u == v for (u, v) in zip(sorted(hashes), sorted(list(set(hashes))))]))

    def test_sse_hash_collisions(self) -> None:
        hashes = []
        for slot_type in ["random", "deterministic"]:
            for heads in [1, 2]:
                for K in [1, 2]:
                    for d in [1, 2]:
                        for h in [2, 4]:
                            for ln_slots in [True, False]:
                                for fixed in [False, True]:
                                    for slot_drop in [0.0, 0.5]:
                                        for attn_act in ["sigmoid", "softmax"]:
                                            for slot_residual in [False, True]:
                                                model = SlotSetEncoder(
                                                    K=K, h=h, d=d, d_hat=h, slot_type=slot_type,
                                                    ln_slots=ln_slots, heads=heads, fixed=fixed, slot_drop=slot_drop,
                                                    attn_act=attn_act, slot_residual=slot_residual
                                                )
                                                hashes.append(md5(str(model)))

        self.assertTrue(len(hashes) == len(list(set(hashes))))
        self.assertTrue(all([u == v for (u, v) in zip(sorted(hashes), sorted(list(set(hashes))))]))

    def test_universal_mbc(self) -> None:
        encoders = [
            partial(MBC, in_dim=16, hidden_dim=16, K=[16], h=[16], d=[16], d_hat=[16], slot_type="deterministic"),
            partial(SetXformerClass, dim_input=16, hidden_dim=16),
            partial(DeepSets, hidden_dim=16, x_dim=16)
        ]
        for set_encoder in encoders:
            for fixed in [False, True]:
                for attn_act in ["softmax", "sigmoid"]:
                    for ln_after in [False, True]:
                        for n_parallel in [1, 2]:
                            umbc = UniversalMBC(
                                set_encoder(), K=16, h=3, d=3,  # type: ignore
                                d_hat=16, heads=4, fixed=fixed,
                                slot_type="deterministic", ln_after=ln_after,
                                n_parallel=n_parallel, attn_act=attn_act
                            )
                            umbc.eval()  # make sure dropout is off

                            x = torch.randn(32, 200, 3)
                            full = umbc(x)

                            for s in x.split(50, dim=1):
                                batched = umbc.process_minibatch(s)

                            
                            passed = check_close(full, batched)
                            if not passed:
                                print(f"max err: {torch.abs(full - batched).amax()=} {set_encoder}")
                            self.assertTrue(passed)

    def test_mbc_slot_set_encoder(self) -> None:
        # THESE ARE FROM THE MBC FILE
        B = 256                     # Batch Size
        n = 200                     # Set Size
        d = 3                       # Element Dimension
        K = 16                      # Number of Slots
        h = 3                      # Slot Size
        d_hat = 32                  # Linear Projection Dimension
        split_size = 50
        X = torch.rand(B, n, d)

        for slot_type in ["random", "deterministic"]:
            for n_heads in [1, 4]:
                for attn_act in ["sigmoid", "slot-sigmoid", "slot-softmax", "slot-exp", "softmax"]:
                    for slot_residual in [False, True]:
                        slot_encoder = SSETester(
                            K=K, h=h, d=d, d_hat=d_hat, slot_type=slot_type,
                            heads=n_heads, attn_act=attn_act, slot_residual=slot_residual
                        )
                        slot_encoder.eval()

                        one = slot_encoder.check_minibatch_consistency(X, split_size=split_size)
                        two = slot_encoder.check_input_invariance(X)
                        three = slot_encoder.check_slot_equivariance(X)
                        self.assertTrue(all([one, two, three]))

    def test_mbc_hierarchical(self) -> None:
        K = [256, 128, 32, 16]      # Heirarchy with reducing number of slots.
        h = [16, 16, 16, 16]        # The dimension of slots in each heirarchy.
        d_hat = [16, 16, 16, 16]    # Projection dimension in each heirarchy
        d = [16, 16, 16, 16]        # Input dimension to each heirarchy
        heads = [4, 4, 4, 4]        # number of heads in each SSE
        split_size = 50             # partition split size
        slot_drops = [0.0, 0.0, 0.0, 0.0]

        def do(*args: Any, **kwargs: Any) -> None:
            torch.manual_seed(1)
            x = torch.rand(256, 200, 16)
            _, dev = x.size(0), x.device

            agg = HierarchicalSSE(*args, **kwargs)
            S = [torch.randn(1, lyr.slots.K, lyr.slots.h, device=dev) for lyr in agg.sse]

            if kwargs["slot_type"] == "random":
                S = [v * lyr.slots.mu + torch.exp(lyr.slots.sigma) for lyr, v in zip(agg.sse, S)]

            enc_full = agg.forward(x, S=S)
            enc_split = agg.forward(x, split_size=split_size, S=S)

            consistency = torch.abs(enc_full - enc_split).amax() < 1e-2
            if not consistency:
                print(f"err: {torch.abs(enc_full - enc_split).amax()}")

            self.assertTrue(consistency)

        with torch.no_grad():
            for pool in ["mean", "sum", "max", "min", "cat", "identity"]:
                do(-1, K, h, d, d_hat, heads, slot_drops, out_dim=4, g=pool, slot_type="random")

            for nheads in [1, 4]:
                do(-1, K, h, d, d_hat, heads, slot_drops, out_dim=4, g="sum", slot_type="random")

            for slot in ["random", "deterministic"]:
                do(-1, K, h, d, d_hat, heads, slot_drops, out_dim=4, g="sum", slot_type=slot)

            for ln_after in [True, False]:
                do(-1, K, h, d, d_hat, heads, slot_drops, out_dim=4, g="sum", slot_type="random", ln_after=ln_after)

    def test_set_xformer_smoketest(self) -> None:
        model = SetXformer(in_dim=3, h_dim=8, out_dim=12, K=4, x_depth=2, d_depth=2)
        inputs = torch.randn(10, 5000, 3)
        out = model(inputs)
        b, k, d = out.size()
        self.assertTrue(all([b == 10, k == 4, d == 12]))

    def test_mbc_set_classification(self) -> None:
        m = MBC(extractor="PermEquiMax")
        x = torch.randn(3, 500, 3)
        out = m(x)
        self.assertEqual(out.size(-1), 40)
        self.assertEqual(len(out.size()), 2)

    def test_set_transformer_set_classification(self) -> None:
        m = SetXformerClass()
        x = torch.randn(3, 500, 3)
        out = m(x)
        self.assertEqual(out.size(-1), 40)
        self.assertEqual(len(out.size()), 2)
        pass

    def test_deepset_classification(self) -> None:
        def do_test(extractor: str = "PermEquiMax", poolfunc="max"):
            model = DeepSets(hidden_dim=32, pool=poolfunc, extractor=extractor)
            inputs = torch.randn(10, 100, 3)
            out = model(inputs)
            self.assertEqual(out.size(-1), 40)
            self.assertEqual(len(out.size()), 2)

        for poolfunc in ["min", "max", "mean", "sum"]:
            do_test(poolfunc=poolfunc)

        for extractor in ["PermEquiMax", "PermEquiMean", "PermEquiMax2", "PermEquiMean2"]:
            do_test(extractor=extractor)
