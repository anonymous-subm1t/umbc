import math
import os
from argparse import ArgumentParser, Namespace
from functools import partial
from typing import Any, Dict, List, Tuple

import torch
from base import Algorithm
from data.get import get_dataset
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm  # type: ignore
from universal_mbc.models.base import HashableModule
from universal_mbc.models.diem.mog_models import EmbedderMoG
from universal_mbc.models.mvn import MBC, DeepSets, SetXformer
from universal_mbc.models.universal_mbc import UniversalMBC
from utils import Stats, md5, seed, set_logger, str2bool

T = torch.Tensor
SetEncoder = nn.Module


class ImageNetTrainer(Algorithm):
    def __init__(self, args: Namespace, model: SetEncoder, trainset: DataLoader, valset: DataLoader):
        super().__init__()

        self.args = args
        self.model = model.to(self.args.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=args.decay_epochs, gamma=args.decay_gamma)
        self.trainset = trainset
        self.valset = valset
        self.epoch = 0
        self.finished = False
        self.tuned = False

        addendum = ""
        self.results_path = os.path.join("results", addendum, f"{trainset.dataset.name}", f"{self.model.name}")  # type: ignore
        self.models_path = os.path.join(self.results_path, "models")
        for d in [self.results_path, self.models_path]:
            os.makedirs(d, exist_ok=True)

        # write the model string to file under the model hash so we will always know which model created this hash
        with open(os.path.join(self.models_path, f"{self.args.model_hash}.txt"), "w") as f:
            f.write(self.args.model_string)

        self.tr_stats = Stats(["loss", "adj_rand_idx"])
        self.te_stats = Stats(["loss", "adj_rand_idx"])

    def fit(self) -> None:
        self.load_model(self.models_path)
        if self.finished:
            self.log("called fit() on a model which has finished training")
            return

        while self.epoch < self.args.epochs:
            # re-initialize sigma and lambda here because during fitting we want to make sure they are
            # blank for the validation phase, but they should be saved in the current model for future testing
            {"train": self.train}[self.args.mode]()
            self.scheduler.step()
            self.log_train_stats(self.results_path)

            if self.epoch % 10 == 0:
                self.test(its=100)
                self.log_test_stats(self.results_path, test_name="val")

            self.save_model(self.models_path)
            self.epoch += 1

        self.log("finished training, load, finish, saving...")
        self.load_model(self.models_path)
        self.save_model(self.models_path, finished=True)

    def nll_and_pred(self, prior: T, mu: T, var: T, x: T) -> Tuple[T, T]:
        b, s, d = x.size()

        logdet = torch.sum(torch.log(var), dim=-1)
        energy = (((mu.unsqueeze(1) - x.view(b, s, 1, d)) ** 2) / var.unsqueeze(1)).sum(dim=-1)  # (b, s, k)
        # print(f"{logdet.size()=} {energy.size()=} {mu.size()=} {var.size()=}")

        nll = 0.5 * var.size(-1) * math.log(2 * math.pi) + 0.5 * logdet.unsqueeze(1) + 0.5 * energy

        log_prior = (prior + 1e-6).log()  # (b, k, 1)
        nll = nll - torch.transpose(log_prior, 1, 2)
        return torch.logsumexp(nll, dim=-1).mean(), nll.argmin(dim=-1)

    def forward(self, x: T) -> Tuple[T, T]:
        # normalize the inputs
        x = (x - x.mean(dim=(1, 2), keepdim=True)) / (x.std(dim=(1, 2), keepdim=True) + 1e-8)

        b, s, d = x.size()
        if all((self.args.slot_drop > 0.0, not self.model.training, self.args.universal, not self.args.fixed_var)):
            # x = (batch, set, d) samples = (samples, batch, K, d)
            samples = self.model.mc(x, samples=self.args.samples)  # type: ignore
            pre_prior, pre_mu, pre_var = torch.softmax(samples[:, :, :, 0], dim=-1).unsqueeze(-1), samples[:, :, :, 1:d + 1], 0.01 + 0.99 * F.softplus(samples[:, :, :, d + 1:])
            prior = pre_prior.mean(dim=0)
            mu = pre_mu.mean(dim=0)
            var = pre_var.mean(dim=0) + pre_mu.var(dim=0)  # (b, k, d)
        elif all((self.args.slot_drop > 0.0, not self.model.training, self.args.universal, self.args.fixed_var)):
            samples = self.model.mc(x, samples=self.args.samples)  # type: ignore
            pre_prior, pre_mu = torch.softmax(samples[:, :, :, 0], dim=-1).unsqueeze(-1), samples[:, :, :, 1:]
            prior = pre_prior.mean(dim=0)
            mu = pre_mu.mean(dim=0)
            var = torch.ones_like(mu) + pre_mu.var(dim=0)  # (b, k, d)
        elif not self.args.fixed_var:
            out = self.model(x)  # out = (batch, K, d), x = (batch, set, d)
            prior, mu, var = torch.softmax(out[:, :, 0], dim=-1).unsqueeze(-1), out[:, :, 1:d + 1], 0.01 + 0.99 * F.softplus(out[:, :, 1 + d:])
        elif self.args.fixed_var:
            out = self.model(x)  # out = (batch, K, d), x = (batch, set, d)
            prior, mu = torch.softmax(out[:, :, 0], dim=-1).unsqueeze(-1), out[:, :, 1:]
            var = torch.ones_like(mu)
        else:
            raise ValueError(f"got an unknown combination: ({self.args=})")

        return self.nll_and_pred(prior, mu, var, x)

    def train(self) -> None:
        self.model.train()
        t = tqdm(self.trainset, ncols=75, leave=False)
        for x, y in t:
            x, y = map(lambda x: x.to(self.args.device), (x, y))  # type: ignore
            _, y = y[:, :self.args.clusters], y[:, self.args.clusters:]

            loss, pred = self.forward(x)

            self.optimizer.zero_grad()
            loss.backward()
            t.set_description(f"{loss.item()=:.4f}")
            self.optimizer.step()

            with torch.no_grad():
                self.tr_stats.update_loss(loss.cpu() * y.numel(), y.numel())
                for _pred, _y in zip(pred.cpu(), y.cpu()):
                    self.tr_stats.update_adj_rand_idx(_pred, _y)

    def test(self, its: int = -1) -> None:
        self.model.eval()
        t = tqdm(self.valset, ncols=75, leave=False)
        with torch.no_grad():
            for i, (x, y) in enumerate(t):
                x, y = map(lambda x: x.to(self.args.device), (x, y))  # type: ignore
                _, y = y[:, :self.args.clusters], y[:, self.args.clusters:]

                loss, pred = self.forward(x)

                self.te_stats.update_loss(loss.cpu() * y.numel(), y.numel())
                for _pred, _y in zip(pred.cpu(), y.cpu()):
                    self.te_stats.update_adj_rand_idx(_pred, _y)

                if its >= 0 and i == its:
                    return

    def get_oracle_benchmark(self) -> None:
        def do_single_instance(_x: T, _y: T, _y_prior: T) -> Tuple[T, T]:
            nlls, preds = [], []  # type: ignore
            for bx, by, bp in zip(x, y, y_prior):  # b is for batch
                mu, var, prior = [], [], []
                for unq_y in torch.unique(by):
                    class_x = bx[by == unq_y]

                    mu.append(class_x.mean(dim=0))
                    var.append(class_x.var(dim=0) + 1e-8)
                    prior.append(bp[int(unq_y.item())])

                nll, pred = self.nll_and_pred(
                    torch.stack(prior).unsqueeze(0).unsqueeze(-1),
                    torch.stack(mu).unsqueeze(0),
                    torch.stack(var).unsqueeze(0),
                    bx.unsqueeze(0)
                )
                nlls.append(nll)
                preds.append(pred)
            return torch.stack(nlls), torch.stack(preds).squeeze(1)

        with torch.no_grad():
            for dset, stats in zip((self.trainset, self.valset), (self.tr_stats, self.te_stats)):
                t = tqdm(dset, ncols=75, leave=False)
                for x, y in t:
                    x, y = map(lambda x: x.to(self.args.device), (x, y))  # type: ignore
                    y_prior, y = y[:, :self.args.clusters], y[:, self.args.clusters:]

                    nll, pred = do_single_instance(x, y, y_prior)

                    stats.update_loss(nll.mean().cpu() * y.numel(), y.numel())
                    for _pred, _y in zip(pred.cpu(), y.cpu()):
                        stats.update_adj_rand_idx(_pred, _y)

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
            "fixed-var": self.args.fixed_var
        }

    def print_log_stats(self, prefix: str, names: List[str], values: List[Any]) -> None:
        msg = f"({prefix}) epoch: {self.epoch}/{self.args.epochs} "
        for _, (n, v) in enumerate(zip(names, values)):
            msg += f"{n}: {v:.4f} "

        self.log(msg)

    def log_train_stats(self, path: str) -> Dict[str, float]:
        result_keys = {**self.get_results_keys(), "run_type": "train"}
        names, values = self.tr_stats.log_stats_df(os.path.join(path, "train-results.csv"), result_keys)
        self.print_log_stats("train", names, values)
        return {n: v for (n, v) in zip(names, values)}

    def log_test_stats(
        self,
        path: str,
        test_name: str = "test",
        additional_keys: Dict[str, Any] = {}
    ) -> Dict[str, float]:
        results_keys = {
            **self.get_results_keys(),
            "run_type": test_name,
            **additional_keys
        }
        names, values = self.te_stats.log_stats_df(os.path.join(path, f"{test_name}-results.csv"), results_keys)
        self.print_log_stats(test_name, names, values)
        return {n: v for (n, v) in zip(names, values)}

    def log(self, msg: str) -> None:
        self.args.logger.info(msg)


class Oracle(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.name = "Oracle"
        self.layer = nn.Linear(1, 1)  # dummy parameter to avoid optimizer error


if __name__ == "__main__":
    parser = ArgumentParser("argument parser for MBC ImageNet Amortized Clustering")

    parser.add_argument("--clusters", type=int, default=4, help="the number of classes to sample in each set")
    parser.add_argument("--comment", type=str, default="", help="comment to add to the hash string and the results file")
    parser.add_argument("--run", type=int, default=0, help="run number")
    parser.add_argument("--epochs", type=int, default=50, help="the number of epochs to run for")
    parser.add_argument("--batch-size", type=int, default=10, help="batch size for training")
    parser.add_argument("--decay-epochs", type=int, nargs="+", default=[35], help="the epochs which learning rate decay occurs")
    parser.add_argument("--decay-gamma", type=float, default=1e-1, help="the learning rate decay rate")
    parser.add_argument("--h-dim", type=int, default=256, help="hidden dim")
    parser.add_argument("--samples", type=int, default=2, help="mc samplse to take for slot drop models")
    parser.add_argument("--n-parallel", type=int, default=1, help="the number of parallel SSE modules to run (universal only)")
    parser.add_argument("--universal", type=str2bool, default=False, help="whether or not to ru the universal MBC model version")
    parser.add_argument("--pooling", type=str, default="mean", choices=["mean", "min", "max", "sum"], help="pooling function if relevant")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--universal-k", type=int, default=128, help="the number of slots to use in the universal encoder")
    parser.add_argument("--gpu", type=int, default=0, help="the gpu index")
    parser.add_argument("--heads", type=int, default=4, help="number of attention heads")
    parser.add_argument("--lr", type=float, default=1e-3, help="initial learning rate")
    parser.add_argument("--mode", type=str, choices=["train", "test", "no-mc-test"])
    parser.add_argument("--slot-type", type=str, default="random", choices=["random", "deterministic"])
    parser.add_argument("--ln-slots", type=str2bool, default=True, help="put layernorm pre-activation on the slots?")
    parser.add_argument("--fixed-var", type=str2bool, default=False, help="whether or not to use fixed variance (identity matrix)")
    parser.add_argument("--ln-after", type=str2bool, default=True, help="put layernorm after SSE")
    parser.add_argument("--fixed", type=str2bool, default=False, help="whether or not to fix the universal SSE weights")
    parser.add_argument("--slot-residual", type=str2bool, default=True, help="whether or not to put a residual connection on the SSE output")
    parser.add_argument("--slot-drop", type=float, default=0.0, help="slot dropout rate for the universal MBC models")
    parser.add_argument("--model", type=str, choices=["mbc", "deepsets", "set-xformer", "oracle", "diff-em"])
    parser.add_argument("--attn-act", type=str, choices=["sigmoid", "softmax", "slot-softmax", "slot-exp"], default="softmax")

    args = parser.parse_args()
    args.logger = set_logger("INFO")
    args.device = torch.device(f"cuda:{args.gpu}")
    args.dataset = "imagenet-clusters"
    args.num_workers = 8

    if args.model == "diff-em":
        args.decay_epochs = []

    # seed before doing anything else with the dataloaders
    seed(args.run)

    distr_emb_args = {"dh": 128, "dout": 64, "num_eps": 5, "layers1": [args.h_dim], "nonlin1": 'relu', "layers2": [args.h_dim], "nonlin2": 'relu'}

    train_ldr, _, test_ldr = get_dataset(args)
    dim = 512
    out_dim = 1 + (dim * (2 if not args.fixed_var else 1))
    in_dim = dim if not args.universal else args.h_dim
    model_deref = {
        "deepsets": partial(DeepSets, in_dim, args.h_dim, out_dim, args.clusters, 3, 3, args.pooling),
        "mbc": partial(
            MBC, in_dim=in_dim, h_dim=args.h_dim, out_dim=out_dim, K=args.clusters,
            x_depth=3, d_depth=3, ln_slots=args.ln_slots, ln_after=args.ln_after,
            slot_type=args.slot_type, attn_act=args.attn_act, slot_residual=args.slot_residual
        ),
        "set-xformer": partial(SetXformer, in_dim, args.h_dim, out_dim, args.clusters, 3, 1, ln=True, isab_enc=False),
        "oracle": partial(Oracle),
        "diff-em": partial(
            EmbedderMoG, dim_input=in_dim, num_outputs=args.clusters, out_type="select_best2",
            num_proto=args.clusters, num_ems=3 if not args.universal else 1,
            dim_feat=args.h_dim, num_heads=5, tau=1e-2,
            distr_emb_args=distr_emb_args, set_out_size=512, net_type="imagenet"
        )
    }
    model = model_deref[args.model]()  # type: ignore

    if args.universal:
        model = UniversalMBC(
            model, K=args.universal_k, h=dim, d=dim, d_hat=args.h_dim,
            slot_type=args.slot_type, ln_slots=args.ln_slots,
            ln_after=args.ln_after, heads=args.heads, fixed=args.fixed,
            slot_drop=args.slot_drop, attn_act=args.attn_act,
            slot_residual=args.slot_residual, n_parallel=args.n_parallel,
            embedder=False
        )

    # hash the string of the model to get a unique identifier which can be used to save the models
    args.model_string = f"COMMENT: {args.comment}\nRUN: {args.run}\n\n" + str(model)
    args.model_hash = md5(args.model_string)
    trainer = ImageNetTrainer(args, model, train_ldr, test_ldr)

    if args.model == "oracle":
        trainer.get_oracle_benchmark()
        trainer.log_train_stats(trainer.results_path)
        trainer.log_test_stats(trainer.results_path, test_name="test")
        exit("finished oracle benchmark")

    if args.mode == "train":
        trainer.fit()
        trainer.log("finished training")
    elif args.mode == "test":
        trainer.args.samples = 100
        trainer.load_model(trainer.models_path)
        test_name = "test"
        if not trainer.finished:
            test_name == "unfinished-model-test"

        trainer.test()
        trainer.log_test_stats(trainer.results_path, test_name=test_name)
    elif args.mode == "no-mc-test":
        trainer.load_model(trainer.models_path)
        test_name = args.mode
        if not trainer.finished:
            test_name == "unfinished-model-" + args.mode

        if not args.universal:
            raise ValueError("this is only applicable to the universal models")

        # set the model sse to have slot drop 0.0
        model.set_drop_rate(0.0)
        args.slot_drop = 0.0

        trainer.test()

        model.set_drop_rate(0.5)  # set it back to 0.5 to keep the stats files consistent
        args.slot_drop = 0.5

        trainer.log_test_stats(
            trainer.results_path,
            test_name=test_name,
            additional_keys={"samples": 0}
        )
    else:
        raise NotImplementedError()
