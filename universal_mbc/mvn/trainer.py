import os
from argparse import ArgumentParser, Namespace
from functools import partial
from typing import Any, Dict, Iterator, List, Tuple

import matplotlib  # type: ignore
import numpy as np
import pandas as pd  # type: ignore
import seaborn as sns  # type: ignore
import torch
from base import Algorithm
from data.get import get_dataset
from matplotlib import animation  # type: ignore
from matplotlib import pyplot as plt  # type: ignore
from matplotlib.animation import FuncAnimation  # type: ignore
from scipy.stats import multivariate_normal  # type: ignore
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from tqdm import tqdm  # type: ignore
from universal_mbc.models.base import HashableModule
from universal_mbc.models.diem.mog_models import EmbedderMoG
from universal_mbc.models.layers.mbc import HierarchicalSSE
from universal_mbc.models.mvn import MBC, DeepSets, SetXformer
from universal_mbc.models.universal_mbc import UniversalMBC
from universal_mbc.plot import set_sns
from utils import Stats, md5, seed, set_logger, str2bool

T = torch.Tensor
SetEncoder = nn.Module


class MVNTrainer(Algorithm):
    def __init__(self, args: Namespace, model: SetEncoder, dataset: Dataset):
        super().__init__()

        self.args = args
        self.model = model.to(self.args.device)

        self.optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=args.decay_epochs, gamma=args.decay_gamma)
        self.trainset = dataset
        self.best_nll = float("inf")
        self.epoch = 0
        self.finished = False
        self.tuned = False

        addendum = ""
        self.results_path = os.path.join("results", addendum, f"{dataset.name}", f"{self.model.name}")  # type: ignore
        self.models_path = os.path.join(self.results_path, "models")
        for d in [self.results_path, self.models_path]:
            os.makedirs(d, exist_ok=True)

        # write the model string to file under the model hash so we will always know which model created this hash
        with open(os.path.join(self.models_path, f"{self.args.model_hash}.txt"), "w") as f:
            f.write(self.args.model_string)

        self.tr_stats = Stats(["loss"])
        self.full_te_stats = [Stats(["loss"]) for _ in self.args.test_set_sizes]
        self.sub_te_stats = [Stats(["loss"]) for _ in self.args.test_set_sizes]

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
                self.test(self.args.val_its)
                self.log_test_stats(self.results_path, test_name="val")

            self.save_model(self.models_path)
            self.epoch += 1

        self.log("finished training, load, finish, saving...")
        self.load_model(self.models_path)
        self.test(self.args.test_its)
        self.log_test_stats(self.results_path, test_name="test")
        self.save_model(self.models_path, finished=True)

    def train(self) -> None:
        self.model.train()
        t = tqdm(range(self.args.epoch_its), ncols=75, leave=False)
        for i in t:
            N = np.random.randint(self.args.set_size // 2, self.args.set_size + 1)
            x, y, pi, mu, sigma = self.trainset.sample(self.args.batch_size, N, self.args.classes)  # type: ignore
            x, y, pi, mu, sigma = map(lambda v: v.to(self.args.device), (x, y, pi, mu, sigma))

            out = self.model(x)

            ll, yhat = self.trainset.log_prob(x, *self.trainset.parse(out))  # type: ignore
            loss = -ll

            self.optimizer.zero_grad()
            loss.backward()

            t.set_description(f"{loss.item()=:.4f}")
            self.optimizer.step()

            with torch.no_grad():
                self.tr_stats.update_loss(loss * y.numel(), y.numel())

    def test_forward(self, x: T) -> T:
        if self.args.slot_drop > 0.0 and self.args.universal:
            # (samples, batch, K, d)
            samples = self.model.mc(x, samples=self.args.samples)  # type: ignore
            mu_prior_out = samples[:, :, :, :3].mean(dim=0)
            sigma_out = samples[:, :, :, 3:].mean(dim=0) + samples[:, :, :, 1:3].var(dim=0)  # law of total variance E[var] + Var[E]
            out = torch.cat((mu_prior_out, sigma_out), dim=-1)
            return out

        return self.model(x)

    def get_sampling_strategies(self) -> List[Any]:
        # define our sampling strategies. We will need to go through the batch one by one and yield
        # different set sample for each set. This can then be iterated over for each set
        def single_point_stream(x: T, y: T) -> Iterator[Tuple[T, T]]:
            """simply sample one element of each set until we reach them end"""
            for idx in torch.randperm(x.size(0)):
                yield x[idx].unsqueeze(0), y[idx].unsqueeze(0)

        def class_stream(x: T, y: T) -> Iterator[Tuple[T, T]]:
            """return each class as a single sample"""
            for cl in y.unique():
                idx = y == cl
                yield x[idx], y[idx]

        def get_chunk_stream(n: int) -> Any:
            def chunk_stream(x: T, y: T) -> Iterator[Tuple[T, T]]:
                """simply sample one element of each set until we reach them end"""
                for idx in torch.randperm(x.size(0)).chunk(n):
                    yield x[idx], y[idx]

            return chunk_stream

        def one_each_stream(x: T, y: T) -> Iterator[Tuple[T, T]]:
            """
            separate each class and then return one instanc of each class until all instances are covered.
            classes will have different numbers of instances, so if oen class runs out of examples, just return the other classes
            """
            xy_lst: List[Tuple[T, T]] = []
            for cl in y.unique():
                idx = y == cl
                xy_lst.append((x[idx], y[idx]))

            max_size = max([v[0].size(0) for v in xy_lst])
            for i in range(max_size):
                out_x, out_y = [], []
                for _x, _y in xy_lst:
                    if i > _x.size(0) - 1:
                        continue
                    out_x.append(_x[i])
                    out_y.append(_y[i])
                yield torch.stack(out_x), torch.stack(out_y)

        return [single_point_stream, class_stream, one_each_stream, get_chunk_stream(128)]

    def motivation_example(self, x: T, y: T, pi: T, mu: T, sigma: T, its: int = 1) -> None:  # type: ignore
        self.model.eval()
        matplotlib.use('Agg')

        outpath = os.path.join("results", f"{self.trainset.name}", "plots", "motivation")  # type: ignore
        os.makedirs(outpath, exist_ok=True)

        set_sns()
        # plt.style.use('seaborn-white')
        cm = sns.color_palette("mako", as_cmap=True)
        sampling_strategies = self.get_sampling_strategies()
        stats = [Stats(["loss"]) for _ in sampling_strategies]

        x, y, pi, mu, sigma = map(lambda v: v.to(self.args.device), (x, y, pi, mu, sigma))
        with torch.no_grad():
            for n, (sx, sy) in enumerate(zip(x, y)):
                xlimit = (sx[:, 0].amin().cpu() - 1, sx[:, 0].amax().cpu() + 1)
                ylimit = (sx[:, 1].amin().cpu() - 1, sx[:, 1].amax().cpu() + 1)

                if n < 2:
                    continue
                if n == 3:
                    return

                if self.args.universal:
                    S = self.model.sse[0].sample_s()  # type: ignore

                # reset the model for this mini batch processing. split it into parts according to the sampling strategies,
                # and save the successive predictions and parts to make final plots and gifs
                for strategy, _ in zip(sampling_strategies, stats):
                    print(f"({n}) {self.model.name} {strategy.__name__}")  # type: ignore
                    self.model.reset()  # type: ignore
                    full_out, model_outs, x_parts, y_parts = torch.Tensor(), [], [], []
                    for part_x, part_y in strategy(sx, sy):
                        part_x, part_y = map(lambda t: t.unsqueeze(0).to(self.args.device), (part_x, part_y))  # add the batch dimension back in

                        x_parts.append(part_x.squeeze(0))
                        y_parts.append(part_y.squeeze(0))
                        if self.args.universal:
                            # sample the slots, MBC test the universal model
                            model_outs.append(self.model.process_minibatch(part_x, S=S).cpu())  # type: ignore
                            if full_out.numel() == 0:
                                full_out = self.model(sx.unsqueeze(0), S=S).cpu()
                            continue
                        model_outs.append(self.model.process_minibatch(part_x).cpu())  # type: ignore

                    # if full_out.numel() != 0 and self.args.universal:
                    #     print(f"diff between full and mbc {strategy.__name__} {self.model.name}: {torch.sum(torch.abs((full_out - model_outs[-1])))}")

                    # make a seaborn plot witht the last prediction and all the x points as a scatterplot
                    # use the model_outs and x_parts to make an animation of the performance at each step
                    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 6))
                    filename = f"{n}-{strategy.__name__}-{self.model.name}"
                    data = pd.DataFrame({
                        "x": sx[:, 0].cpu().numpy().tolist(),
                        "y": sx[:, 1].cpu().numpy().tolist(),
                        "label": [str(v) for v in sy.cpu().numpy().tolist()],
                    })
                    ax = sns.scatterplot(data=data, x="x", y="y", hue="label", style="label", ax=ax, s=11**2)

                    outs = model_outs[-1].squeeze(0)
                    ll, _ = self.trainset.log_prob(sx.unsqueeze(0).cpu(), *self.trainset.parse(model_outs[-1]))  # type: ignore

                    def contours(preds: Any, steps: int = 100) -> Any:
                        conts = []
                        for i, g in enumerate(preds):
                            stds = 20
                            mu, sigma = g[1:3], torch.clamp(F.softplus(g[3:]), 1e-2)

                            mx = np.linspace(-sigma[0].item() * stds, sigma[0].item() * stds, steps)
                            my = np.linspace(-sigma[1].item() * stds, sigma[1].item() * stds, steps)
                            xx, yy = np.meshgrid(mx, my)

                            rv = multivariate_normal([0, 0], [[sigma[0].item(), 0], [0, sigma[1].item()]])
                            data = np.dstack((xx, yy))
                            z = rv.pdf(data)
                            c = ax.contour(mx + mu[0].cpu().numpy(), my + mu[1].cpu().numpy(), z, 5, alpha=0.75, linewidths=4.0, cmap=cm)
                            conts.append(c)
                        return conts

                    contours(outs, steps=100)
                    # data = pd.DataFrame({"x": x_samples, "y": y_samples, "labels": [str(v)for v in labels]})
                    # ax = sns.kdeplot(data=data, x="x", y="y", hue="labels", palette="hls")

                    model_deref = {"SetXformer": "Set Transformer", "universal-SetXformer": "UMBC + Set Transformer"}
                    ax.set(xlabel="", ylabel="", xticks=[], yticks=[], ylim=ylimit, xlim=xlimit)
                    # ax.set_title(f"{model_deref[self.model.name]}\n{' '.join(strategy.__name__.split('_'))}", fontsize=24, fontweight="bold")  # type: ignore

                    ax.text(x=-1.25, y=3.25, s=f"NLL$\downarrow$: {-ll:.2f}", fontsize=24, fontweight="bold", ha="center", va="center")  # type: ignore
                    ax.text(x=-1.25, y=-2.5, s=f"{model_deref[self.model.name]}\n{' '.join(strategy.__name__.split('_'))}", fontsize=24, fontweight="bold", ha="center", va="center")  # type: ignore
                    ax.legend(fontsize=24, markerscale=2.0)

                    fig.tight_layout()
                    # fig.savefig(os.path.join(outpath, f"{filename}.png"))
                    fig.savefig(os.path.join(outpath, f"{filename}.pdf"))
                    plt.clf()
                    plt.cla()
                    plt.close()

                    if True:
                        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 6))
                        model_name = self.model.name
                        colors = ["tab:blue", "tab:orange", "tab:green", "tab:purple"]
                        sc = [ax.scatter(x=[], y=[], c=c, label=i, s=11**2) for i, c in enumerate(colors)]
                        x_parts = [v.cpu().numpy() for v in x_parts]
                        x_parts_done = [[] for _ in colors]  # type: ignore

                        class Animator:
                            def __init__(self, trainset: Dataset) -> None:
                                self.trainset = trainset

                            def update(self, frame: Any) -> Any:
                                i, ax = frame
                                print(".", end="", flush=True)

                                labels = y_parts[i].cpu().numpy()
                                for lbl in np.unique(labels):
                                    idx = labels == lbl
                                    # print(x_parts[:i + 1])

                                    x_parts_done[lbl].append(x_parts[i][idx])
                                    sc[lbl].set_offsets(np.concatenate(x_parts_done[lbl]))
                                    # sc[lbl].set_offsets(np.c_[x_parts[i][idx, 0].cpu().numpy(), x_parts[i][idx, 1].cpu().numpy()])
                                    # ax.scatter(x=x_parts[i][idx, 0].cpu().numpy(), y=x_parts[i][idx, 1].cpu().numpy(), c=colors[lbl], label=lbl, s=11**2)

                                ll, _ = self.trainset.log_prob(sx.unsqueeze(0).cpu(), *self.trainset.parse(model_outs[i]))  # type: ignore
                                if hasattr(self, "text"):
                                    self.text.remove()  # type: ignore

                                ax.set_title(f"{model_deref[model_name]} ({i}) {' '.join(strategy.__name__.split('_'))}", fontsize=18)  # type: ignore
                                self.text = ax.text(x=-1.25, y=3.25, s=f"NLL$\downarrow$: {-ll:.2f}", fontsize=24, fontweight="bold", ha="center", va="center")  # type: ignore
                                ax.set(xlabel="", ylabel="", xticks=[], yticks=[], xlim=xlimit, ylim=ylimit)

                                if hasattr(self, "conts"):
                                    for coll in [c.collections for c in self.conts]:  # type: ignore
                                        for tp in coll:
                                            tp.remove()

                                self.conts = contours(model_outs[i].squeeze(0), steps=200)
                                out = []
                                for cont in self.conts:
                                    out += cont.collections
                                return out

                            def frames(self) -> Iterator[Any]:
                                for i in range(len(model_outs)):
                                    yield i, ax

                        engine = Animator(trainset=self.trainset)
                        interval = 1000 if strategy.__name__ == "class_stream" else 100
                        fps = 1 if strategy.__name__ == "class_stream" else 30
                        ani = FuncAnimation(
                            fig, engine.update, frames=engine.frames, interval=interval, blit=True, save_count=len(x_parts)
                        )

                        fig.tight_layout()
                        writergif = animation.PillowWriter(fps=fps)
                        ani.save(os.path.join(outpath, f"animation-{filename}.gif"), writer=writergif)
                        plt.clf()
                        plt.cla()
                        plt.close()

    def test(self, its: int) -> None:  # type: ignore
        self.model.eval()
        t = tqdm(range(its), ncols=75, leave=False)
        for it in t:
            x, y, pi, mu, sigma = self.trainset.sample(self.args.batch_size, self.args.test_set_size, self.args.classes)  # type: ignore
            x, y, pi, mu, sigma = map(lambda v: v.to(self.args.device), (x, y, pi, mu, sigma))  # type: ignore

            with torch.no_grad():
                for i, ss in enumerate(self.args.test_set_sizes):
                    xss = x[:, :ss]
                    out = self.test_forward(xss)

                    full_ll, yhat = self.trainset.log_prob(x, *self.trainset.parse(out))  # type: ignore
                    self.full_te_stats[i].update_loss(-full_ll * y.numel(), y.numel())

                    sub_ll, yhat = self.trainset.log_prob(xss, *self.trainset.parse(out))  # type: ignore
                    self.sub_te_stats[i].update_loss(-sub_ll * y.numel(), y.numel())

    def load_model(self, path: str) -> None:
        model_path = os.path.join(path, f"{self.args.model_hash}.pt")
        if os.path.exists(model_path):
            saved = torch.load(model_path, map_location="cpu")
            self.epoch = saved["epoch"]
            self.best_nll = saved["best_nll"]

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
            best_nll=self.best_nll,
            tuned=self.tuned
        )
        torch.save(save, sd_path)

    def get_results_keys(self, additional_keys: Dict[str, Any] = {}) -> Dict[str, Any]:
        return {
            **(self.model.get_results_keys() if isinstance(self.model, HashableModule) else {"model": self.model.name}),  # type: ignore
            "epoch": self.epoch,
            "model_hash": self.args.model_hash,
            "run": self.args.run,
            "comment": self.args.comment,
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
        additional_keys: Dict[str, Any] = {}
    ) -> Dict[str, float]:
        for i, ss in enumerate(self.args.test_set_sizes):
            results_keys = {
                **self.get_results_keys(),
                "run_type": test_name,
                "train_set_size": self.args.set_size,
                "test_set_size": ss,
                "ref_set_size": args.test_set_sizes[-1],
                **additional_keys,  # for adding extra keys for special tests
            }
            _, _ = self.full_te_stats[i].log_stats_df(os.path.join(path, f"{test_name}-results.csv"), results_keys)

            results_keys = {**results_keys, "ref_set_size": ss}  # overwrite the ref set size for this version which contains smaller ref set
            names, values = self.sub_te_stats[i].log_stats_df(os.path.join(path, f"{test_name}-results.csv"), results_keys)
            self.print_log_stats(f"{test_name}-{ss}", names, values)

        return {n: v for (n, v) in zip(names, values)}

    def log(self, msg: str) -> None:
        self.args.logger.info(msg)


if __name__ == "__main__":
    parser = ArgumentParser("argument parser for MBC MVN")

    parser.add_argument("--dataset", type=str, default="toy-mixture-of-gaussians", choices=["toy-mixture-of-gaussians"], help="the dataset to use")
    parser.add_argument("--mvn-type", type=str, default="diag", choices=["full", "diag"], help="type of covariance for MoG dataset")
    parser.add_argument("--comment", type=str, default="", help="comment to add to the hash string and the results file")
    parser.add_argument("--run", type=int, default=0, help="run number")
    parser.add_argument("--epoch-its", type=int, default=1000, help="the number of iters to run in an epoch")
    parser.add_argument("--test-its", type=int, default=1000, help="the number of iters to run in test mode")
    parser.add_argument("--val-its", type=int, default=200, help="the number of iters to run in test mode")
    parser.add_argument("--epochs", type=int, default=50, help="the number of epochs to run for")
    parser.add_argument("--samples", type=int, default=10, help="the number of mc samples to take for slot dropout testing")
    parser.add_argument("--batch-size", type=int, default=10, help="batch size for training")
    parser.add_argument("--decay-epochs", type=int, nargs="+", default=[35], help="the epochs which learning rate decay occurs")
    parser.add_argument("--decay-gamma", type=float, default=1e-1, help="the learning rate decay rate")
    parser.add_argument("--classes", type=int, default=4, help="classes")
    parser.add_argument("--set-size", type=int, default=16, help="batch size for training")
    parser.add_argument("--h-dim", type=int, default=128, help="hidden dim")
    parser.add_argument("--test-set-sizes", type=int, nargs="+", default=[8, 16, 32, 64, 128, 256, 512, 1024], help="for testing effect of set size at test time")
    parser.add_argument("--universal", type=str2bool, default=False, help="whether or not to ru the universal MBC model version")
    parser.add_argument("--test-set-size", type=int, default=1024, help="full set size of testing")
    parser.add_argument("--pooling", type=str, default="mean", choices=["mean", "min", "max", "sum"], help="pooling function if relevant")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--universal-k", type=int, default=128, help="the number of slots to use in the universal encoder")
    parser.add_argument("--n-parallel", type=int, default=1, help="the number of parallel SSE modules to run (universal only)")
    parser.add_argument("--gpu", type=int, default=0, help="the gpu index")
    parser.add_argument("--heads", type=int, default=4, help="number of attention heads")
    parser.add_argument("--lr", type=float, default=1e-3, help="initial learning rate")
    parser.add_argument("--mode", type=str, choices=["train", "test", "mc-ablation-test", "mbc-motivation-example"])
    parser.add_argument("--slot-type", type=str, default="random", choices=["random", "deterministic"])
    parser.add_argument("--ln-slots", type=str2bool, default=True, help="put layernorm pre-activation on the slots?")
    parser.add_argument("--ln-after", type=str2bool, default=True, help="put layernorm after SSE")
    parser.add_argument("--fixed", type=str2bool, default=False, help="whether or not to fix the universal SSE weights")
    parser.add_argument("--slot-residual", type=str2bool, default=True, help="whether or not to put a residual connection on the SSE output")
    parser.add_argument("--slot-drop", type=float, default=0.0, help="slot dropout rate for the universal MBC models")
    parser.add_argument("--model", type=str, choices=["mbc", "deepsets", "set-xformer", "diff-em", "stacked-sse"])
    parser.add_argument("--attn-act", type=str, choices=["sigmoid", "softmax", "slot-sigmoid", "slot-softmax", "slot-exp"], default="the attention activation on MBC models")

    args = parser.parse_args()
    args.logger = set_logger("INFO")
    args.device = torch.device(f"cuda:{args.gpu}")
    args.dim = 2

    # seed before doing anything else with the dataloaders
    seed(args.run)

    distr_emb_args = {"dh": 128, "dout": 64, "num_eps": 5, "layers1": [args.h_dim], "nonlin1": 'relu', "layers2": [args.h_dim], "nonlin2": 'relu'}

    train_ldr, _, _ = get_dataset(args)

    if args.mode == "mbc-motivation-example":
        x, y, pi, mu, sigma = train_ldr.dataset.sample(args.batch_size * 10, args.test_set_size, args.classes)  # type: ignore

    out_dim = 1 + 2 + (2 if args.mvn_type == "diag" else 4)
    in_dim = 2 if not args.universal else args.h_dim
    model_deref = {
        "deepsets": partial(DeepSets, in_dim, args.h_dim, out_dim, args.classes, 2, 2, args.pooling),
        "mbc": partial(
            MBC, in_dim=in_dim, h_dim=args.h_dim, out_dim=out_dim, K=args.classes,
            x_depth=2, d_depth=2, ln_slots=args.ln_slots, ln_after=args.ln_after,
            slot_type=args.slot_type, attn_act=args.attn_act, slot_residual=args.slot_residual
        ),
        "stacked-sse": partial(
            HierarchicalSSE,
            in_dim=2,
            K=[128, 64, 32, 16, 8, 4],
            h=[args.h_dim for _ in range(6)],
            d=[args.h_dim for _ in range(6)],
            d_hat=[args.h_dim for _ in range(6)],
            heads=[args.heads for _ in range(6)],
            slot_drops=[args.slot_drop for _ in range(5)] + [0.0],  # last layer needs to not drop any slots for prediction
            g="identity",
            attn_act=args.attn_act,
            slot_residual=args.slot_residual,
            ln_after=args.ln_after,
            slot_type=args.slot_type,
        ),
        "set-xformer": partial(SetXformer, in_dim, args.h_dim, out_dim, args.classes, 2, 2, ln=True),
        "diff-em": partial(
            EmbedderMoG, dim_input=in_dim, num_outputs=args.classes, out_type="select_best2",
            num_proto=args.classes, num_ems=3 if not args.universal else 1,
            dim_feat=args.h_dim, num_heads=5, tau=1e-2,
            distr_emb_args=distr_emb_args, set_out_size=2
        )
    }

    model = model_deref[args.model]()  # type: ignore
    if args.universal:
        model = UniversalMBC(
            model, K=args.universal_k, h=2, d=2, d_hat=args.h_dim,
            slot_type=args.slot_type, ln_slots=args.ln_slots,
            ln_after=args.ln_after, heads=args.heads, fixed=args.fixed,
            slot_drop=args.slot_drop, attn_act=args.attn_act,
            slot_residual=args.slot_residual, n_parallel=args.n_parallel,
            embedder=True
        )

    # hash the string of the model to get a unique identifier which can be used to save the models
    args.model_string = f"TRAIN SET SIZE: {args.set_size}\nCOMMENT: {args.comment}\nRUN: {args.run}\n\n" + str(model)
    args.model_hash = md5(args.model_string)

    trainer = MVNTrainer(args, model, train_ldr.dataset)

    if args.mode == "train":
        trainer.fit()
        trainer.log("finished training")
    elif args.mode == "test":
        trainer.load_model(trainer.models_path)
        trainer.test(its=1000)
        trainer.log_test_stats(trainer.results_path)
    elif args.mode == "mbc-motivation-example":
        if args.model != "set-xformer":
            raise ValueError("this experiment is only meant for the set transformer (both universal and not)")

        trainer.load_model(trainer.models_path)
        trainer.motivation_example(x, y, pi, mu, sigma)

    elif args.mode == "mc-ablation-test":
        if not args.universal:
            raise ValueError("cannot run mc ablation test on non unveral module")

        trainer.load_model(trainer.models_path)

        #  reset the dropout rate of the model to 0 after loading it.
        orig_slot_drop = args.slot_drop
        trainer.model.set_drop_rate(0.0)  # type: ignore
        args.slot_drop = 0.0

        # run the test to see what happens when a dropout model does not mc test
        trainer.test(its=1000)
        trainer.log_test_stats(
            trainer.results_path,
            test_name=args.mode,
            additional_keys={"samples": 0}
        )

        # run the test for 100, 1000, 1000 mc samples to see the effect of many mc samples
        trainer.model.set_drop_rate(orig_slot_drop)  # type: ignore
        for samples in [10, 100, 1000, 10000]:
            args.samples = samples
            trainer.test(its=1000)
            trainer.log_test_stats(
                trainer.results_path,
                test_name=args.mode,
                additional_keys={"samples": samples}
            )
    else:
        raise NotImplementedError()
