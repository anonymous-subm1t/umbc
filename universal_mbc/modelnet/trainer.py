import os
from argparse import ArgumentParser, Namespace
from functools import partial
from typing import Any, Dict, List

import torch
from base import Algorithm
from data.get import get_dataset
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm  # type: ignore
from universal_mbc.models.base import HashableModule
from universal_mbc.models.classification import (MBC, DeepSets, SetXformer,
                                                 clip_grad)
from universal_mbc.models.diem.mog_models import EmbedderMoG
from universal_mbc.models.universal_mbc import UniversalMBC
from utils import Stats, md5, seed, set_logger, str2bool

T = torch.Tensor
SetEncoder = nn.Module


class ModelNetTrainer(Algorithm):
    def __init__(self, args: Namespace, model: SetEncoder, train: DataLoader, val: DataLoader):
        super().__init__()

        self.args = args
        self.model = model.to(self.args.device)

        self.optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=args.decay_step, gamma=args.decay_gamma)
        self.trainset = train
        self.clipper = clip_grad if (self.args.model in ["deepsets"] and not args.universal) else lambda model, max_norm: 0.0
        self.valset = val
        self.epoch = 0
        self.finished = False
        self.tuned = False

        addendum = ""
        if args.slot_drop == 0.0:
            addendum = "no-drop"
        self.results_path = os.path.join("results", addendum, f"{train.dataset.name}", f"{self.model.name}")  # type: ignore
        self.models_path = os.path.join(self.results_path, "models")
        for d in [self.results_path, self.models_path]:
            os.makedirs(d, exist_ok=True)

        # write the model string to file under the model hash so we will always know which model created this hash
        with open(os.path.join(self.models_path, f"{self.args.model_hash}.txt"), "w") as f:
            f.write(self.args.model_string)

        self.tr_stats = Stats(["accuracy", "nll", "ece"])
        self.te_stats = [Stats(["accuracy", "nll", "ece"]) for _ in self.args.test_set_sizes]

    def fit(self) -> None:
        self.load_model(self.models_path)
        if self.finished:
            self.log("called fit() on a model which has finished training")
            return

        while self.epoch <= self.args.epochs:
            # re-initialize sigma and lambda here because during fitting we want to make sure they are
            # blank for the validation phase, but they should be saved in the current model for future testing
            {"train": self.train}[self.args.mode]()
            self.log_train_stats(self.results_path)
            self.scheduler.step()

            if self.epoch % 10 == 0:
                self.test()
                self.log_test_stats(self.results_path, test_name="val")

            self.save_model(self.models_path)
            self.epoch += 1

        self.log("finished training, load, finish, saving...")
        self.load_model(self.models_path)
        self.save_model(self.models_path, finished=True)

    def train(self) -> None:
        self.model.train()
        t = tqdm(self.trainset, ncols=75, leave=False)
        for i, (x, y) in enumerate(t):
            x, y = map(lambda v: v.to(self.args.device), (x, y))

            logit = self.model(x)
            loss = F.cross_entropy(logit, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.clipper(self.model, 5)
            self.optimizer.step()

            t.set_description(f"{loss.item()=:.4f}")
            with torch.no_grad():
                self.tr_stats.update_acc((logit.argmax(dim=-1) == y).sum().item(), y.size(0))
                self.tr_stats.update_nll(logit, y)
                self.tr_stats.update_ece(logit, y)

    def test_forward(self, x: T) -> T:
        if self.args.slot_drop > 0.0 and self.args.universal:
            # (samples, batch, K)
            samples = self.model.mc(x, samples=self.args.samples)  # type: ignore
            return samples.softmax(dim=-1).mean(dim=0)
        return self.model(x).softmax(dim=-1)

    def test(self) -> None:
        self.model.eval()
        t = tqdm(self.valset, ncols=75, leave=False)
        with torch.no_grad():
            for (x, y) in t:
                x, y = map(lambda v: v.to(self.args.device), (x, y))
                sample_idx = torch.randperm(x.size(1))  # sample this noise outside of the loop so every set size gets new data
                for i, ss in enumerate(self.args.test_set_sizes):
                    xss = x[:, sample_idx[:ss]]
                    pred = self.test_forward(xss)

                    self.te_stats[i].update_acc((pred.argmax(dim=-1) == y).sum().item(), y.size(0))
                    self.te_stats[i].update_nll(pred, y, softmaxxed=True)
                    self.te_stats[i].update_ece(pred, y, softmaxxed=True)

    def load_model(self, path: str) -> None:
        model_path = os.path.join(path, f"{self.args.model_hash}.pt")
        if os.path.exists(model_path):
            saved = torch.load(model_path, map_location="cpu")
            self.epoch = saved["epoch"]

            self.model.load_state_dict(saved["state_dict"])
            self.model = self.model.to(self.args.device)
            self.optimizer.load_state_dict(saved["optimizer"])
            self.scheduler.load_state_dict(saved["scheduler"])

            self.finished = saved["finished"]
            self.tuned = saved.get("tuned", False)
            print(f"loaded saved model: {self.epoch=} {self.finished=}\nfrom path: {model_path}")

    def save_model(self, path: str, finished: bool = False) -> None:
        sd_path = os.path.join(path, f"{self.args.model_hash}.pt")
        save = dict(
            epoch=self.epoch,
            state_dict=self.model.state_dict(),
            optimizer=self.optimizer.state_dict(),
            scheduler=self.scheduler.state_dict(),
            finished=finished,
            tuned=self.tuned
        )
        torch.save(save, sd_path)

    def get_results_keys(self) -> Dict[str, Any]:
        return {
            **(self.model.get_results_keys() if isinstance(self.model, HashableModule) else {"model": self.model.name}),  # type: ignore
            "epoch": self.epoch,
            "model_hash": self.args.model_hash,
            "run": self.args.run,
            "comment": self.args.comment,
            "samples": self.args.samples
        }

    def print_log_stats(self, prefix: str, names: List[str], values: List[Any]) -> None:
        msg = f"({prefix}) epoch: {self.epoch}/{self.args.epochs} "
        for _, (n, v) in enumerate(zip(names, values)):
            msg += f"{n}: {v:.4f} "

        self.log(msg)

    def log_train_stats(self, path: str) -> Dict[str, float]:
        result_keys = {**self.get_results_keys(), "run_type": "train", "train_set_size": self.args.set_size}
        names, values = self.tr_stats.log_stats_df(os.path.join(path, "train-results.csv"), result_keys)
        self.print_log_stats("train", names, values)
        return {n: v for (n, v) in zip(names, values)}

    def log_test_stats(
        self,
        path: str,
        test_name: str = "test",
        extra_keys: Dict[str, Any] = {}
    ) -> Dict[str, float]:
        for i, ss in enumerate(self.args.test_set_sizes):
            results_keys = {
                **self.get_results_keys(),
                "run_type": test_name,
                "train_set_size": self.args.set_size,
                "test_set_size": ss,
                **extra_keys
            }
            names, values = self.te_stats[i].log_stats_df(os.path.join(path, f"{test_name}-results.csv"), results_keys)
            self.print_log_stats(f"{test_name}-{ss}", names, values)
        return {n: v for (n, v) in zip(names, values)}

    def log(self, msg: str) -> None:
        self.args.logger.info(msg)


if __name__ == "__main__":
    parser = ArgumentParser("argument parser for MBC ModelNet40")

    parser.add_argument("--dataset", type=str, default="modelnet-2048", help="the dataset to load")
    parser.add_argument("--data-root", type=str, default="/d1/dataset/ModelNet40-2048/data", help="the path to the datasets folder")
    parser.add_argument("--num-workers", type=int, default=8, help="run number")
    parser.add_argument("--comment", type=str, default="", help="comment to add to the hash string and the results file")
    parser.add_argument("--run", type=int, default=0, help="run number")
    parser.add_argument("--samples", type=int, default=1, help="the number of mc samples to take for slot dropout testing")
    parser.add_argument("--n-parallel", type=int, default=1, help="the number of parallel SSE modules to run (universal only)")
    parser.add_argument("--epochs", type=int, default=1000, help="the number of epochs to run for")
    parser.add_argument("--batch-size", type=int, default=64, help="batch size for training")
    parser.add_argument("--decay-step", type=int, nargs="+", default=[400, 800], help="the epochs which learning rate decay occurs")
    parser.add_argument("--decay-gamma", type=float, default=1e-1, help="the learning rate decay rate")
    parser.add_argument("--h-dim", type=int, default=128, help="hidden dim")
    parser.add_argument("--set-size", type=int, default=100, help="set size for training")
    parser.add_argument("--test-set-sizes", type=int, nargs="+", default=[100, 1000, 2048], help="for testing effect of set size at test time")
    parser.add_argument("--test-set-size", type=int, default=2048, help="full set size of testing")
    parser.add_argument("--universal", type=str2bool, default=False, help="whether or not to the universal MBC model version")
    parser.add_argument("--pooling", type=str, default="max", choices=["mean", "min", "max", "sum"], help="pooling function if relevant")
    parser.add_argument("--weight-decay", type=float, default=1e-7)
    parser.add_argument("--universal-k", type=int, default=128, help="the number of slots to use in the universal encoder")
    parser.add_argument("--gpu", type=int, default=0, help="the gpu index")
    parser.add_argument("--heads", type=int, default=4, help="number of attention heads")
    parser.add_argument("--lr", type=float, default=1e-3, help="initial learning rate")
    parser.add_argument("--mode", type=str, choices=["train", "test", "corrupt-test", "mc-drop-ablation-test"])
    parser.add_argument("--slot-type", type=str, default="random", choices=["random", "deterministic"])
    parser.add_argument("--ln-slots", type=str2bool, default=True, help="put layernorm pre-activation on the slots?")
    parser.add_argument("--ln-after", type=str2bool, default=True, help="put layernorm after SSE")
    parser.add_argument("--fixed", type=str2bool, default=False, help="whether or not to fix the universal SSE weights")
    parser.add_argument("--slot-residual", type=str2bool, default=True, help="whether or not to put a residual connection on the SSE output")
    parser.add_argument("--slot-drop", type=float, default=0.0, help="randomly drops slots with probability p")
    parser.add_argument("--model", type=str, choices=["mbc", "deepsets", "set-xformer", "diff-em", "stacked-sse"])
    parser.add_argument("--attn-act", type=str, choices=["sigmoid", "softmax", "slot-softmax", "slot-exp"], default="softmax")

    args = parser.parse_args()
    args.logger = set_logger("INFO")
    args.device = torch.device(f"cuda:{args.gpu}")

    # seed before doing anything else with the dataloaders
    seed(args.run)
    distr_emb_args = {"dh": args.h_dim, "dout": 64, "num_eps": 5, "layers1": [args.h_dim], "nonlin1": 'relu', "layers2": [args.h_dim], "nonlin2": 'relu'}

    trainset, _, testset = get_dataset(args)

    in_dim = 3 if not args.universal else args.h_dim
    model_deref = {
        "deepsets": partial(DeepSets, hidden_dim=args.h_dim, x_dim=in_dim),
        "mbc": partial(
            MBC, in_dim=in_dim, hidden_dim=args.h_dim, ln_slots=args.ln_slots,
            ln_after=args.ln_after, slot_type=args.slot_type,
            slot_residual=args.slot_residual, attn_act=args.attn_act,
            slot_drops=[0.0], K=[16], h=[args.h_dim], d=[args.h_dim],  # slot drop is only for the universal module
            d_hat=[args.h_dim]
        ),
        "set-xformer": partial(SetXformer, dim_input=in_dim, hidden_dim=args.h_dim),
        "diff-em": partial(
            EmbedderMoG, dim_input=in_dim, num_outputs=1, out_type="select_best2",
            num_proto=16, num_ems=2, dim_feat=args.h_dim,
            num_heads=1, tau=1e-3, set_out_size=40, net_type="modelnet"  # use the same number of heads used for the SSE slots
        )
    }
    model = model_deref[args.model]()  # type: ignore

    if args.universal:
        model = UniversalMBC(
            model, K=args.universal_k, h=3, d=3, d_hat=args.h_dim,
            slot_type=args.slot_type, ln_slots=args.ln_slots,
            ln_after=args.ln_after, heads=args.heads, fixed=args.fixed,
            slot_drop=args.slot_drop, attn_act=args.attn_act,
            slot_residual=args.slot_residual, n_parallel=args.n_parallel,
            embedder=True
        )

    # hash the string of the model to get a unique identifier which can be used to save the models
    args.model_string = f"TRAIN SET SIZE: {args.set_size}\nCOMMENT: {args.comment}\nRUN: {args.run}\n\n" + str(model)
    args.model_hash = md5(args.model_string)
    trainer = ModelNetTrainer(args, model, trainset, testset)

    if args.mode == "train":
        trainer.fit()
        trainer.log("finished training")
    elif args.mode == "test":
        trainer.args.samples = 10
        trainer.load_model(trainer.models_path)
        test_name = "test"
        if not trainer.finished:
            test_name == "unfinished-model-test"

        trainer.test()
        trainer.log_test_stats(trainer.results_path, test_name=test_name)
    elif args.mode == "mc-drop-ablation-test":
        trainer.load_model(trainer.models_path)
        if not trainer.finished:
            raise ValueError(f"trainer should be finished to run this test: got ({trainer.finished=})")

        orig_slot_drop = args.slot_drop
        trainer.model.set_drop_rate(0.0)  # type: ignore
        trainer.args.slot_drop = 0.0
        trainer.args.samples = 0

        # run the test to see what happens when a dropout model does not mc test
        trainer.test()
        trainer.log_test_stats(trainer.results_path, test_name=args.mode, extra_keys={"test_drop_rate": 0.0})

        # run the test for 100, 1000, 1000 mc samples to see the effect of many mc samples
        trainer.model.set_drop_rate(orig_slot_drop)  # type: ignore
        for rate in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            for samples in [5, 10, 25, 50, 100, 250, 500]:
                if rate <= 0.5 and samples <= 100:
                    continue  # skipping because we already have data for these

                trainer.args.samples = samples
                trainer.args.slot_drop = rate
                trainer.model.set_drop_rate(rate)  # type: ignore

                trainer.test()
                trainer.log_test_stats(
                    trainer.results_path,
                    test_name=args.mode,
                    extra_keys={"test_drop_rate": rate}
                )
    elif args.mode == "corrupt-test":
        args.dataset = "modelnet-2048-c"

        # in the corrupted sets, some of the corruptions add points, so just set it to a high number to make sure we sample them all
        trainer.args.test_set_sizes = [100, 1000, 5000]
        trainer.args.samples = 100

        trainer.load_model(trainer.models_path)
        test_name = "corrupt-test"
        if not trainer.finished:  # safeguard to keep unfinished results firewalled from the finished model
            test_name == "unfinished-model-test"

        # load the original dataset, set the dataset, and run the uncorrupted test
        trainer.args.corruption = "original"
        trainer.args.severity = 0
        _, valset, testset = get_dataset(trainer.args)
        trainer.valset = testset

        trainer.test()
        trainer.log_test_stats(
            trainer.results_path,
            test_name=test_name,
            extra_keys={"corruption": "original", "severity": 0}
        )

        corruptions = [
            "background", "cutout", "density", "density_inc", "distortion",
            "distortion_rbf", "distortion_rbf_inv", "gaussian", "impulse", "lidar",
            "occlusion", "rotation", "shear", "uniform", "upsampling"
        ]
        for corr in corruptions:
            for severity in [1, 2, 3, 4, 5]:
                # load the right dataset
                trainer.log(f"running test for {corr=} {severity=}")
                trainer.args.corruption = corr
                trainer.args.severity = severity
                _, valset, testset = get_dataset(trainer.args)

                # set the right dataset
                trainer.valset = testset

                # run the test
                trainer.test()
                trainer.log_test_stats(
                    trainer.results_path,
                    test_name=test_name,
                    extra_keys={"corruption": corr, "severity": severity}
                )
    else:
        raise NotImplementedError()
