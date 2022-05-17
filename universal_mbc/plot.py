import os.path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd  # type: ignore
import seaborn as sns  # type: ignore
from matplotlib import pyplot as plt  # type: ignore
from utils import get_module_root

j = os.path.join
Q = Dict[str, Any]


def set_sns() -> None:
    sns.set_theme(style="white")
    sns.color_palette("tab10")
    sns.set_context(
        "notebook",
        font_scale=1.7,
        rc={
            "lines.linewidth": 3,
            "lines.markerscale": 4,
        }
    )


def remove_ax_legend_title(ax: Any) -> None:
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels)


def get_dataframes(base_path: str, filename: str = "test-results.csv") -> Tuple[pd.DataFrame, List[str]]:
    # call this here so anything plotting a dtaframe will have the same setup
    set_sns()

    dataframes, models = [], []
    for model in os.listdir(base_path):
        if model == "plots":
            continue  # skip the plot directory

        results_path = j(base_path, model, filename)
        if not os.path.exists(results_path):
            print(f"WARNING: {results_path=} does not exist")
            continue

        print(f"loading: {results_path=}")
        df = pd.read_csv(results_path, sep=",")
        df["model"] = model

        dataframes.append(df)
        models.append(model)

    df = pd.concat(dataframes)
    return df, models


def drop_duplicates_old_this_needs_to_be_refactored(df: pd.DataFrame) -> pd.DataFrame:
    """
    needs to be refactored because this was made before knowing that the index we need to slice out
    will change dependent on the experiment. This is still ok for the MVN experiment
    """
    # slice out the timestamp column and remove all duplicated rows. This happens
    # when the test set may be run twice. For the MVN experiment there may be randomness
    # by running on differetn GPU's, so also eliminate the loss column
    duplicate_cols = df.columns[:1].tolist() + df.columns[3:].tolist()
    df = df.drop_duplicates(subset=duplicate_cols)
    return df


# if the arg is none, then we do not want to query that result and instead we will want
# to make some kind of plot along that dimension
def df_query(df: pd.DataFrame, **kwargs: Q) -> pd.DataFrame:
    for k in kwargs:
        if kwargs[k] is not None:
            df = df[df[k] == kwargs[k]]

    return df


def filestr(s: str) -> str:
    return "-".join(s.split(" ")).lower()


def plot_dropout_times() -> None:
    set_sns()
    root = get_module_root()
    arr = np.load(os.path.join(root, "universal_mbc", "plots", "drop-times.npy"))

    data = {"p": [], "time": []}  # type: ignore
    for i, arr_p in enumerate(arr):
        data["p"].extend([1 / (i + 1) for _ in range(arr_p.shape[0])])
        data["time"].extend(arr_p.tolist())

    df = pd.DataFrame(data)

    fig, ax = plt.subplots(nrows=1, ncols=1)
    sns.lineplot(data=df, x="p", y="time")

    ax.set_title("Inference Speed vs. Slot Dropout Rate", fontsize=18)
    fig.tight_layout()
    fig.savefig(os.path.join(root, "universal_mbc", "plots", "drop-times.pdf"))
    fig.savefig(os.path.join(root, "universal_mbc", "plots", "drop-times.png"))


def plot_embedding_ll() -> None:
    set_sns()
    root = get_module_root()
    data = pd.read_csv(os.path.join(root, "universal_mbc", "plots", "st-umbcst-embeddings.csv"))
    arr = np.load(os.path.join(root, "universal_mbc", "plots", "st-umbcst-embeddings.npy"))

    new_data = {"mean_variance": [], "chunk_size": [], "model": []}  # type: ignore
    for universal in [False, True]:
        for size in [2, 4, 8, 16, 32, 64]:
            idx = ((data.chunk_size == size) & (data.universal == universal)).tolist()
            new_arr = arr[idx]

            mean_var = np.var(new_arr, axis=0)

            for var in mean_var:
                new_data["mean_variance"].append(var)
                new_data["chunk_size"].append(size)
                new_data["model"].append("UMBC+Set Transformer" if universal else "Set Transformer")

            # print(new_data)

    df = pd.DataFrame(new_data)
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax = sns.lineplot(data=df, x="chunk_size", y="mean_variance", hue="model", ax=ax, marker="o", ci=68)
    ax.legend(fontsize=17)
    ax.set(xlabel="chunk size", ylabel="mean $\sigma^2$")
    ax.set_title("mini-batch pooling $\sigma^2$")
    fig.tight_layout()
    fig.savefig(os.path.join(root, "universal_mbc", "plots", "embedding-var.pdf"))
    fig.savefig(os.path.join(root, "universal_mbc", "plots", "embedding-var.png"))


if __name__ == "__main__":
    plot_dropout_times()
    # plot_embedding_ll()
