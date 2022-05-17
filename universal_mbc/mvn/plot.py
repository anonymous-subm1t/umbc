import copy
import os.path
from typing import List

import pandas as pd  # type: ignore
import seaborn as sns  # type: ignore
from matplotlib import pyplot as plt  # type: ignore
from universal_mbc.plot import (
    Q, df_query, drop_duplicates_old_this_needs_to_be_refactored, filestr,
    get_dataframes, remove_ax_legend_title)

j = os.path.join
BASE_RESULTS_PATH = j("results", "MixtureOfGaussians-diag")

outpath = j(BASE_RESULTS_PATH, "plots")
os.makedirs(outpath, exist_ok=True)


model_deref = {
    "MBC": "Slot Set Encoder",
    "DiffEM": "Diff. EM",
    "DeepSets": "Deep Sets",
    "SetXformer": "Set Transformer",
    "universal-SetXformer": "UMBC + Set Transformer",
    "universal-DiffEM": "UMBC + Diff. EM",
    "universal-DeepSets": "UMBC + DeepSets",
    "universal-MBC": "UMBC + Slot Set Encoder",
}


def get_sse_kwargs(variant: str = "UNIVERSAL", k: int = 128) -> Q:
    prefix = {"UNIVERSAL": "sse.0", "MBC": "encoder.0", "HIERARCHICAL": "sse.0"}[variant]
    return {
        "train_set_size": 512,
        "test_set_size": 512,
        "ref_set_size": 1024,
        ":ln_after": True,
        f"{prefix}:attn_act": "softmax",
        f"{prefix}.slots:slot_type": "random",
        f"{prefix}:heads": 4,
        f"{prefix}:slot_drop": 0.0,
        f"{prefix}:slot_residual": True,
        f"{prefix}.slots:K": k,
        f"{prefix}.slots:fixed": False
    }


DEFAULT_ST_KWARGS: Q = {
    "train_set_size": 512,
    "test_set_size": 512,
    "ref_set_size": 1024,
}

DEFAULT_DS_KWARGS: Q = DEFAULT_ST_KWARGS
DEFAULT_DIEM_KWARGS: Q = DEFAULT_ST_KWARGS


def get_default_kwargs(model: str) -> Q:
    if "universal" in model:  # this case must go first
        return {**get_sse_kwargs("UNIVERSAL"), ":n_parallel": 1}
    elif "Hierarchical" in model:
        return get_sse_kwargs("HIERARCHICAL")
    elif "Transformer" in model or "Xformer" in model:
        return DEFAULT_ST_KWARGS
    elif "DeepSets" in model:
        return DEFAULT_DS_KWARGS
    elif "DiffEM" in model:
        return DEFAULT_DIEM_KWARGS
    else:
        return get_sse_kwargs("MBC", k=4)


def print_best(df: pd.DataFrame, models: List[str]) -> None:
    for model in models:
        model_df = df[(df.model == model)]
        best_ll_idx = model_df[(model_df.test_set_size == 512)]["loss"].idxmin(axis=1)
        best_row = model_df.iloc[best_ll_idx]

        print(f"best settings for {model=}", end=" ")
        for title, col in zip(model_df.columns, best_row):
            print(f"{title}={col}", end="\n")
        print("\n\n")


def plot_mc_ablation_test(df: pd.DataFrame, models: List[str]) -> None:
    for key, title, filename, xlabel, ylim in zip(["samples"], ["Slot Dropout Samples"], ["mc-dropout-ablation"], ["Test Set Size"], [(1.75, 1.9)]):
        fig, ax = plt.subplots(nrows=1, ncols=1)

        if len(models) > 1:
            raise ValueError("cannot have more than one model for this plot")

        # copy default kwargs and set this key to None so we query each possible value
        query = copy.deepcopy(get_sse_kwargs("UNIVERSAL"))
        query["sse.0:slot_drop"] = None
        query["test_set_size"] = None
        query[key] = None
        query["ref_set_size"] = 1024

        model_df = df_query(df, **query)  # type: ignore
        model_df = drop_duplicates_old_this_needs_to_be_refactored(model_df)
        model_df = model_df.astype({"samples": str})

        print(model_df["test_set_size"].unique())

        ax = sns.lineplot(data=model_df, x="test_set_size", y="loss", hue="samples", ax=ax, marker="o", ci=68)
        ax.set(title=title, xlabel=xlabel, ylabel="NLL", ylim=ylim)
        remove_ax_legend_title(ax)

        fig.tight_layout()
        fig.savefig(j(outpath, f"{filestr(filename)}.png"))
        fig.savefig(j(outpath, f"{filestr(filename)}.pdf"))
        plt.close()


def plot_line_ablation_attn_heads(df: pd.DataFrame, models: List[str]) -> None:
    for key, title, filename, xlabel in zip(["sse.0:heads"], ["Number of Attention Heads"], ["universal-heads-vs-parallel"], ["Heads"]):

        fig, ax = plt.subplots(nrows=1, ncols=1)
        all_df = []
        for model in models:
            # copy default kwargs and set this key to None so we query each possible value
            print(f"{model=} {key=}")
            query = copy.deepcopy(get_sse_kwargs("UNIVERSAL"))
            query[key] = None
            query[":n_parallel"] = None
            query["ref_set_size"] = 512

            model_df = df[(df.model == model)]
            model_df = df_query(model_df, **query)  # type: ignore
            model_df = drop_duplicates_old_this_needs_to_be_refactored(model_df)
            all_df.append(model_df)
            print(model_df.loc[:, ["loss", "sse.0:heads", "sse.0.slots:K"]])

        all_df = pd.concat(all_df).rename(columns={":n_parallel": "num-parallel"})
        ax = sns.lineplot(data=all_df, x=key, y="loss", hue="num-parallel", ax=ax, marker="o", ci=68)
        ax.set(title=title, xlabel=xlabel, ylabel="NLL")

        fig.tight_layout()
        fig.savefig(j(outpath, f"{filestr(filename)}.png"))
        fig.savefig(j(outpath, f"{filestr(filename)}.pdf"))
        plt.close()


def plot_line_ablation(df: pd.DataFrame, models: List[str]) -> None:
    for key, title, filename, xlabel in zip(
        ["sse.0:heads", "sse.0.slots:K", ":n_parallel"],
        ["Number of Attention Heads", "Number of Slots", "Number of Parallel Unviersal Blocks"],
        ["universal-heads", "universal-slots", "universal-parallel"],
        ["Heads", "Slots", "Blocks"]
    ):

        fig, ax = plt.subplots(nrows=1, ncols=1)
        all_df = []
        for model in models:
            # copy default kwargs and set this key to None so we query each possible value
            print(f"{model=} {key=}")
            query = copy.deepcopy(get_sse_kwargs("UNIVERSAL"))
            query[key] = None
            query["ref_set_size"] = 512

            model_df = df[(df.model == model)]
            model_df = df_query(model_df, **query)  # type: ignore
            model_df = drop_duplicates_old_this_needs_to_be_refactored(model_df)
            all_df.append(model_df)
            print(model_df.loc[:, ["loss", "sse.0:heads", "sse.0.slots:K"]])

        all_df = pd.concat(all_df)
        ax = sns.lineplot(data=all_df, x=key, y="loss", hue="model", ax=ax, marker="o", ci=68, legend=False)
        ax.set(title=title, xlabel=xlabel, ylabel="NLL")
        # remove_ax_legend_title(ax)

        fig.tight_layout()
        fig.savefig(j(outpath, f"{filestr(filename)}.png"))
        fig.savefig(j(outpath, f"{filestr(filename)}.pdf"))
        plt.close()


def plot_categorical_ablation_by_test_set_size(df: pd.DataFrame, models: List[str]) -> None:
    keys = ["sse.0:attn_act", "sse.0.slots:slot_type", "sse.0:slot_drop", "sse.0:slot_residual", ":ln_after", "sse.0.slots:fixed"]
    titles = ["Attention Activations", "Slot Type", "Slot Dropout", "Slot Residual", "FF Layernorm", "Fixed Slots"]
    filenames = ["attn-acts", "slot-types", "slot-drop", "slot-residual", "ln-after", "fixed"]
    ylims = [(1.725, 1.90), (1.7, 1.9), (1.7, 2.0), (1.7, 1.95), (1.75, 1.85), (1.5, 5.0)]

    for key, title, filename, ylim in zip(keys, titles, filenames, ylims):
        fig, ax = plt.subplots(nrows=1, ncols=1)
        all_df = []
        for model in models:
            # copy default kwargs and set this key to None so we query each possible value
            # we will plot these performances by set size, so also set test set size to None
            print(f"{model=} {key=}")
            query = copy.deepcopy(get_sse_kwargs("UNIVERSAL"))
            query[key] = None
            query["test_set_size"] = None
            query["ref_set_size"] = 1024

            model_df = df[(df.model == model)]
            model_df = df_query(model_df, **query)  # type: ignore
            model_df = drop_duplicates_old_this_needs_to_be_refactored(model_df)
            model_df = model_df.astype({key: str})
            all_df.append(model_df)
            print(model_df.loc[:, ["loss", "sse.0:heads", "sse.0.slots:K", key]])

        all_df = pd.concat(all_df)
        sns.lineplot(data=all_df, x="test_set_size", y="loss", hue=key, ax=ax, marker="o", ci=68)
        ax.set(title=title, ylim=ylim, xlabel="test set size", ylabel="NLL")
        remove_ax_legend_title(ax)

        fig.tight_layout()
        fig.savefig(j(outpath, f"{filestr(filename)}.png"))
        fig.savefig(j(outpath, f"{filestr(filename)}.pdf"))
        plt.close()


def set_size_lineplot(df: pd.DataFrame, models: List[str]) -> None:
    # make pairs of models which can be zipped up. Everything should be in the form
    # [model, ..., universal-model, ...] so we can just cut the list in half and zip
    models = [v for v in models if "Hierarc" not in v]
    models.sort()
    limits = ((2.62, 2.72), (1.6, 2.0), (1.95, 2.2), (1.6, 1.9), (2.62, 2.72), (1.8, 2.0), (1.95, 2.2), (1.7, 1.9))
    for mdl, (ymin, ymax) in zip(models, limits):
        fig, ax = plt.subplots(nrows=1, ncols=1)

        # copy default kwargs and set this key to None so we query each possible value
        # we will plot these performances by set size, so also set test set size to None
        query = copy.deepcopy(get_default_kwargs(mdl))

        def query_all_sets(q: Q) -> Q:
            q["test_set_size"] = None
            q["ref_set_size"] = 1024
            q["train_set_size"] = None
            return q

        query = query_all_sets(query)
        model_df = df[(df.model == mdl)]

        print(f"before query: {model_df=}")
        print(f"{query=}")
        model_df = df_query(model_df, **query)
        model_df = drop_duplicates_old_this_needs_to_be_refactored(model_df)
        model_df = model_df.astype({"train_set_size": str})
        print(f"after query: {model_df=}")

        sns.lineplot(data=model_df, x="test_set_size", y="loss", hue="train_set_size", ax=ax, marker="o", ci=68)

        ax.set(ylim=(ymin, ymax), title=model_deref[mdl], xlabel="test set size", ylabel="NLL")
        legend = ax.legend(title="train set size")
        if mdl != "DeepSets":
            legend.remove()

        fig.tight_layout()
        fig.savefig(os.path.join(outpath, f"{mdl.lower()}-line.png"))
        fig.savefig(os.path.join(outpath, f"{mdl.lower()}-line.pdf"))
        plt.close()


if __name__ == "__main__":
    df, models = get_dataframes(BASE_RESULTS_PATH)

    # print_best(df, models)
    # plot_line_ablation(df, [m for m in models if ("universal-SetXformer" in m)])
    plot_line_ablation_attn_heads(df, [m for m in models if ("universal-SetXformer" in m)])
    # plot_categorical_ablation_by_test_set_size(df, [m for m in models if ("universal-SetXformer" in m)])
    # set_size_lineplot(df, models)

    # df, models = get_dataframes(BASE_RESULTS_PATH, filename="mc-ablation-test-results.csv")
    # plot_mc_ablation_test(df, models)
