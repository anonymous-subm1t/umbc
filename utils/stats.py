import os
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import torch
from sklearn.metrics import adjusted_rand_score  # type: ignore
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.distributions import Normal

from utils import ece_partial, ece_partial_final

__all__ = ["Stats"]


T = torch.Tensor


class Stats:
    """
    Stats is used to track stats during both training and validation.

    There are many common stat types in machine learning which are calculated in similar ways, so
    this is an abtract class to track any type of stat with a few values as kwargs. Once an 'update'
    function and a 'crunch' function are defined, everything else should just work as expected
    """

    stat_attributes = {
        "correct": 0.0,
        "acc_total": 0.0,
        "loss": 0.0,
        "loss_total": 0.0,
        "ll": 0.0,
        "nll": 0.0,
        "nll_total": 0.0,
        "accs": 0.0,
        "confs": 0.0,
        "n_in_bins": 0.0,
        "n": 0.0,
        "auroc": 0.0,
        "adj_rand_idx": 0.0,
        "adj_rand_idx_total": 0.0,
        "aupr": 0.0,
        "softmax_entropy": 0.0,
        "softmax_entropy_total": 0.0,
        "y_true": [],
        "y_score": [],

        # stats for regression
        "mu": [],
        "sigma": [],
        "y": [],
    }

    logs = ["id_ood_entropy"]

    # stats are things which can be tracked throughout training and then logged once at the end of training
    stats = ["accuracy", "loss", "nll", "ece", "aupr", "auroc", "reg_ece", "reg_nll", "mse", "softmax_entropy", "adj_rand_idx"]

    def __init__(self, stats: List[str], logs: List[Tuple[str, str]] = None) -> None:
        for s in stats:
            if s not in self.stats:
                raise ValueError(f"stat: {s} needs to be one of: {self.stats}")

        self.stats_tracked = stats
        self.crunch_funcs = {
            "accuracy": self.crunch_accuracy,
            "loss": self.crunch_loss,
            "nll": self.crunch_nll,
            "ece": self.crunch_ece,
            "aupr": self.crunch_aupr,
            "softmax_entropy": self.crunch_softmax_entropy,
            "adj_rand_idx": self.crunch_adj_rand_idx,
        }
        self.zero()

        # if the logs arleady exist for a previous run, we should overwrite the file with a new blank file
        # which has the current timestamp. Later when we write wto the log with this class, we will append to
        # the file created here.
        self.logs_tracked = {}
        if logs is not None:
            for (log, file) in logs:
                if log not in self.logs:
                    raise ValueError(f"{log=} is invalid (choices: {self.logs})")

                # create the directory path if it does not already exist
                path = os.path.split(file)[0]
                os.makedirs(path, exist_ok=True)

                with open(file, "w") as _:
                    pass

                # save the path under the log name so we can update the lofgile later
                self.logs_tracked[log] = file

        self.crunched = False

    def zero(self) -> None:
        self.crunched = False

        for att in self.stat_attributes:
            if isinstance(self.stat_attributes[att], list):
                setattr(self, att, [])
                continue
            setattr(self, att, self.stat_attributes[att])

    def set(self, attrs: List[Tuple[str, Any]]) -> None:
        for (name, val) in attrs:
            setattr(self, name, val)

    def crunch(self) -> None:
        if not self.crunched:
            for stat_name in self.stats_tracked:
                if stat_name == "auroc":
                    continue  # skip because this is included in aupr
                self.crunch_funcs[stat_name]()
            self.crunched = True

    def print(self) -> None:
        """print all the stats without logging them anywhere"""
        self.crunch()
        values = [getattr(self, v) for v in self.stats_tracked]
        names = [v for v in self.stats_tracked]

        for (n, v) in zip(names, values):
            print(f"{n}: {v:0.4f} ", end=" ")

    def get_stats(self) -> Dict[str, Any]:
        self.crunch()
        values = [getattr(self, v) for v in self.stats_tracked]
        names = [v for v in self.stats_tracked]

        out = {}
        for (k, v) in zip(names, values):
            out[k] = v
        return out

    def log_stats(self, path: str) -> Tuple[List[Any], ...]:
        self.crunch()

        if not os.path.exists(path):
            # make the directory path if it does not already exist
            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path), exist_ok=True)

            with open(path, "w+") as f:
                f.write(f"{','.join([v for v in self.stats_tracked])}\n")

        values = [getattr(self, v) for v in self.stats_tracked]
        names = [v for v in self.stats_tracked]
        with open(path, "a+") as f:
            f.write(f"{','.join([str(v) for v in values])}\n")

        self.zero()
        return names, values

    def log_stats_df(self, path: str, info_dict: Dict[str, Any]) -> Tuple[List[Any], ...]:
        """
        this was made as an experimental new way to log stats using a dataframe instead
        of a manually created csv file
        """
        self.crunch()

        values = [getattr(self, v) for v in self.stats_tracked]
        names = [v for v in self.stats_tracked]
        # we are appending a single row, but all of the columns need to be in a list
        data: Dict[str, Any] = {n: [v] for (n, v) in zip(names, values)}
        data["timestamp"] = str(datetime.now())
        for k in info_dict:
            data[k] = [info_dict[k]]

        os.makedirs(os.path.dirname(path), exist_ok=True)
        df = pd.DataFrame(data)
        with open(path, 'a+') as f:
            df.to_csv(f, mode='a', header=f.tell() == 0)  # only write header if the current line is 0

        # if os.path.exists(path):
        #     old_df = pd.read_csv(path)
        #     df = pd.concat(old_df, df)

        self.zero()
        return names, values

    def update_aupr_auroc(self, y_true: T, y_score: T) -> None:
        self.y_true.append(y_true.detach().cpu().long())  # type: ignore
        self.y_score.append(y_score.detach().cpu())  # type: ignore

    def crunch_aupr(self) -> None:
        self.y_true = torch.cat(self.y_true)  # type: ignore
        self.y_score = torch.cat(self.y_score)  # type: ignore

        # average precision score is only for the multiclass setting, so only use this if the
        # y_score has a larger class dimension.
        if len(self.y_score.size()) > 1 and self.y_score.size(1) > 1:
            y_one_hot = torch.zeros((self.y_true.size(0), self.y_score.size(1)))
            y_one_hot[torch.arange(y_one_hot.size(0)), self.y_true] = 1
            self.aupr = average_precision_score(y_one_hot, self.y_score)
            self.auroc = roc_auc_score(y_one_hot, self.y_score)
            return

        self.aupr = average_precision_score(self.y_true, self.y_score)
        self.auroc = roc_auc_score(self.y_true, self.y_score)

    def update_acc(self, correct: int, n: int) -> None:
        self.correct += correct  # type: ignore
        self.acc_total += n  # type: ignore

    def crunch_accuracy(self) -> None:
        self.accuracy = self.correct / self.safe_denom(self.acc_total)  # type: ignore

    def update_adj_rand_idx(self, yhat: T, y: T) -> None:
        self.adj_rand_idx += abs(adjusted_rand_score(y, yhat))  # type: ignore
        self.adj_rand_idx_total += 1  # type: ignore

    def crunch_adj_rand_idx(self) -> None:  # type: ignore
        self.adj_rand_idx = self.adj_rand_idx / self.safe_denom(self.adj_rand_idx_total)  # type: ignore

    def update_softmax_entropy(self, logits: T, n: int, softmaxxed: bool = False) -> None:
        if not softmaxxed:
            logits = logits.softmax(dim=-1)

        logits = torch.clamp(logits, 1e-45)
        self.softmax_entropy += -(logits * torch.log(logits)).sum(dim=-1).sum().item()  # type: ignore
        self.softmax_entropy_total += n  # type: ignore

    def crunch_softmax_entropy(self) -> None:
        self.softmax_entropy = self.softmax_entropy / self.safe_denom(self.softmax_entropy_total)  # type: ignore

    def update_loss(self, loss: T, n: int) -> None:
        self.loss += loss.detach().cpu().item()  # type: ignore
        self.loss_total += n  # type: ignore

    def safe_denom(self, val: float) -> float:
        return val + 1e-10

    def crunch_loss(self) -> None:
        self.loss = self.loss / self.safe_denom(self.loss_total)  # type: ignore

    def update_nll(self, logits: T, y: T, softmaxxed: bool = False) -> None:
        if not softmaxxed:
            logits = logits.softmax(dim=-1)

        logits = torch.clamp(logits, 1e-45).log()

        self.ll += torch.gather(logits.detach().cpu(), 1, y.view(-1, 1).detach().cpu()).sum().item()  # type: ignore
        self.nll_total += y.size(0)  # type: ignore

    def log_id_ood_entropy(self, id_ood_label: T, logits: T, softmaxxed: bool = False) -> None:
        # there is nothing to crunch ofr this one since we just need to store a list of them and then log it later
        if not softmaxxed:
            logits = logits.softmax(dim=-1)

        logits = torch.clamp(logits, 1e-45)

        entropy = -(logits * torch.log(logits)).sum(dim=-1)
        with open(self.logs_tracked["id_ood_entropy"], "a+") as f:
            np.savetxt(f, torch.cat((id_ood_label.unsqueeze(-1).cpu(), entropy.unsqueeze(-1).cpu()), dim=-1).numpy())

    def crunch_nll(self) -> None:
        # it is ok to take the log here because the value in update nll for classification is the softmax probability
        # which needs to be log(.)ed
        self.nll = -self.ll / self.safe_denom(self.nll_total)  # type: ignore

    def update_ece(self, logits: T, y: T, softmaxxed: bool = False) -> None:
        confs, accs, n_in_bins, n = ece_partial(y.detach().cpu(), logits.detach().cpu(), softmaxxed=softmaxxed)
        self.accs += accs  # type: ignore
        self.confs += confs  # type: ignore
        self.n_in_bins += n_in_bins  # type: ignore
        self.n += n  # type: ignore

    def crunch_ece(self) -> None:
        self.ece = ece_partial_final(self.confs, self.accs, self.n_in_bins, self.n)  # type: ignore
