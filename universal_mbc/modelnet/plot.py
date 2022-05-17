import os.path
from typing import List

import pandas as pd  # type: ignore
import seaborn as sns  # type: ignore
from matplotlib import pyplot as plt  # type: ignore
from universal_mbc.plot import Q, df_query, get_dataframes, set_sns
from utils import shift_level_box_plot

j = os.path.join
BASE_RESULTS_PATH = j("results", "ModelNet40-2048")

outpath = j(BASE_RESULTS_PATH, "plots", "boxplots")
os.makedirs(outpath, exist_ok=True)


def get_sse_kwargs(variant: str = "UNIVERSAL", k: int = 32) -> Q:
    prefix = {"UNIVERSAL": "sse.0", "MBC": "ro.0.sse.0"}[variant]
    return {
        f"{prefix}:attn_act": "softmax",
        f"{prefix}.slots:slot_type": "random",
        f"{prefix}:heads": 4,
        # f"{prefix}:slot_drop": 0.5,
        f"{prefix}:slot_drop": 0.0,
        f"{prefix}:slot_residual": True,
        f"{prefix}.slots:K": k,
        f"{prefix}.slots:fixed": False
    }


DEFAULT_ST_KWARGS: Q = {}
DEFAULT_DS_KWARGS: Q = {}
DEFAULT_DIFFEM_KWARGS: Q = {}


def get_default_kwargs(model: str) -> Q:
    if "universal" in model:  # this case must go first
        return {**get_sse_kwargs(variant="UNIVERSAL", ), ":n_parallel": 4}
    elif "Transformer" in model or "Xformer" in model:
        return DEFAULT_ST_KWARGS
    elif "DeepSets" in model:
        return DEFAULT_DS_KWARGS
    elif "DiffEM" in model:
        return DEFAULT_DIFFEM_KWARGS
    elif "Oracle" in model:
        return {}
    else:
        return {**get_sse_kwargs("MBC", k=16), "ro.0.sse.0:slot_drop": 0.0}


# used for renaming models for the official plots
METHODS_TO_RENAMES = {
    "MBC": "SSE",
    "DeepSets": "Deep Sets",
    "SetTransformer": "Set Transformer",
    "DiffEM": "Diff. EM",
    "universal-SetTransformer": "UMBC+Set Transformer",
    "universal-MBC": "UMBC+SSE",
    "universal-DeepSets": "UMBC+Deep Sets",
    "universal-DiffEM": "UMBC+Diff. EM",
}

METHODS_TO_COLORS = {
    "SSE": "tab:gray",
    "Deep Sets": "tab:cyan",
    "Set Transformer": "tab:blue",
    "Diff. EM": "tab:olive",
    "UMBC+Set Transformer": "tab:orange",
    "UMBC+SSE": "tab:purple",
    "UMBC+Deep Sets": "tab:green",
    "UMBC+Diff. EM": "tab:pink",
}

# used to control the hue order in the plot
MODEL_NAMES_ALL_CORRS = [
    "SSE",
    "Deep Sets",
    "Diff. EM",
    "UMBC+Diff. EM",
    "Set Transformer",
    "UMBC+Set Transformer",
]

MODEL_NAMES_INDIVIDUAL = [
    "SSE",
    "UMBC+SSE",
    "Deep Sets",
    "UMBC+Deep Sets",
    "Diff. EM",
    "UMBC+Diff. EM",
    "Set Transformer",
    "UMBC+Set Transformer",
]


METRIC_NAMES = {"accuracy": "Accuracy", "nll": "NLL", "ece": "ECE"}


def print_modelnet_table_values(df: pd.DataFrame, models: List[str]) -> None:
    for model in df.model.unique():
        for metric in ["accuracy", "nll", "ece"]:
            print(f"{model=} {metric=}")
            vals = []
            for test_set_size in [100, 1000, 2048]:
                df_test_ss = df[(df.test_set_size == test_set_size) & (df.model == model)]
                df_test_ss = df_test_ss.sort_values(by="test_set_size")

                query = get_default_kwargs(model)
                df_test_ss = df_query(df_test_ss, **query)

                df_test_ss = df_test_ss[metric].to_numpy()
                if metric in ["accuracy", "ece"]:
                    df_test_ss *= 100

                mu = df_test_ss.mean()
                std = df_test_ss.std()
                vals.append(f"{mu:.2f}$\pm${std:.2f}")

            print(f"{model=} {metric=} {' & '.join(vals)}")


def modelnet_mc_drop_ablation(df: pd.DataFrame, models: List[str]) -> None:
    ablation_path = j(BASE_RESULTS_PATH, "plots", "mc-ablation")
    os.makedirs(ablation_path, exist_ok=True)
    set_sns()
    sns.set_context("notebook", font_scale=4.0)

    raise ValueError("make this function remove duplicates if you have to use it again, there may be double rows since we ran a bigger experiment with more sample ranges")

    df["accuracy"] *= 100
    df["ece"] *= 100

    for model in models:
        for metric in ["accuracy", "nll", "ece"]:
            for test_set_size in [100, 1000, 2048]:  # test set size is 500 because some of the corruptions added points, so we sampled 5000 to be sure we included all points
                print(f"plotting: {metric=} {test_set_size=} {model=}")

                # query this model and this test set size
                model_df = df[(df.model == model) & (df.test_set_size == test_set_size)].copy()
                model_df = model_df.rename(columns={"test_drop_rate": "p"})

                # get the p = 0.0 which is the deterministic version. This should be constant
                # along the top sample row of the matrix
                p_zero = model_df[(model_df.p == 0.0) & (model_df.samples == 0)]

                # remove the p_zero from the model_df, and then cope the p_zero and add
                # it in with every sample size
                model_df = model_df[(model_df.p != 0) & (model_df.samples != 0)]
                for smpls in [5, 10, 25, 50, 100]:
                    tmp = p_zero.copy()
                    tmp.loc[:, "samples"] = smpls
                    model_df = pd.concat((model_df, tmp))

                grouped = model_df.groupby(["p", "samples"])
                model_df.loc[:, "mean"] = grouped[metric].transform("mean")
                model_df.loc[:, "std"] = grouped[metric].transform("std")
                model_df = model_df[(model_df.run == 0)]
                # model_df.loc[:, "display_value"] = [f"{u:.1f}$\pm${v:.1f}" for (u, v) in zip(model_df["mean"].tolist(), model_df["std"].tolist())]
                model_df.loc[:, "display_value"] = [f"{u:.2f}" for (u, v) in zip(model_df["mean"].tolist(), model_df["std"].tolist())]

                pivot_display = model_df.pivot(index="p", columns="samples", values="display_value")
                pivot_val = model_df.pivot(index="p", columns="samples", values="mean")

                fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 12))

                sns.heatmap(
                    pivot_val,
                    annot=pivot_display,
                    fmt="s",
                    ax=ax,
                    annot_kws={"fontsize": 28, "fontweight": "bold"},
                    cmap="mako" if metric != "accuracy" else "mako_r"
                )

                arrow = r"$\uparrow$" if metric == "accuracy" else "$\downarrow$"
                ax.set_title(f"{METRIC_NAMES[metric]} {arrow}", fontsize=48)
                ax.set_xlabel("Samples", fontsize=48)
                ax.set_ylabel("Dropout Rate", fontsize=48)

                fig.tight_layout()
                # fig.savefig(os.path.join(ablation_path, f"{model}-{metric}-{test_set_size}.png"))
                fig.savefig(os.path.join(ablation_path, f"{model}-{metric}-{test_set_size}.pdf"))
                plt.close()


def modelnet_boxplot_all_corrs(df: pd.DataFrame, models: List[str]) -> None:
    df = df.rename(columns={"severity": "level", "model": "method"})
    df = df.replace({"method": METHODS_TO_RENAMES})

    # filter out models which are not in the input args models list
    filtered_df = []
    for mdl in models:
        filtered_df.append(df[df.method == METHODS_TO_RENAMES[mdl]])
    df = pd.concat(filtered_df)

    for metric in ["accuracy", "nll", "ece"]:
        for test_set_size in [100, 1000, 5000]:  # test set size is 500 because some of the corruptions added points, so we sampled 5000 to be sure we included all points
            print(f"plotting: {metric=} {test_set_size=}")

            fig, ax = plt.gcf(), plt.gca()
            fig.set_size_inches(15, 4)

            df_copy = df[df.test_set_size == test_set_size].copy()
            df_copy = df_copy.rename(columns={metric: "value"})
            df_copy.loc[df_copy.level == 0, "level"] = "Test"
            # print(df_copy["level"])

            # we had to call it 5000 in the code because some of the corruptions added points above the normal 2048 so we sampled up to 5000 to be safe
            ss = 2048 if test_set_size == 5000 else test_set_size
            metric_name = metric.capitalize() if metric == "accuracy" else metric.upper()
            shift_level_box_plot(ax, df_copy, metric_name, METHODS_TO_COLORS, f"All Corruptions, Test Set Size: {ss}", hue_order=MODEL_NAMES_ALL_CORRS, legend=ss == 100, legend_loc="upper left")

            fig.tight_layout()
            # fig.savefig(os.path.join(outpath, f"all-corrs-{metric}-{ss}.png"))
            fig.savefig(os.path.join(outpath, f"all-corrs-{metric}-{ss}.pdf"))
            plt.close()


def modelnet_boxplot_individual(df: pd.DataFrame, models: List[str]) -> None:
    df = df.rename(columns={"severity": "level", "model": "method"})
    df = df.replace({"method": METHODS_TO_RENAMES})
    for metric in ["accuracy", "nll", "ece"]:
        for test_set_size in [100, 1000, 5000]:  # test set size is 500 because some of the corruptions added points, so we sampled 5000 to be sure we included all points
            corruptions = [
                "background", "cutout", "density", "density_inc", "distortion",
                "distortion_rbf", "distortion_rbf_inv", "gaussian", "impulse", "lidar",
                "occlusion", "rotation", "shear", "uniform", "upsampling"
            ]
            for corr in corruptions:
                print(f"plotting: {metric=} {test_set_size=} {corr=}")

                fig, ax = plt.gcf(), plt.gca()
                fig.set_size_inches(15, 4)

                df_copy = df[((df.corruption == corr) | (df.corruption == "original")) & (df.test_set_size == test_set_size)].copy()
                df_copy = df_copy.rename(columns={metric: "value"})
                df_copy.loc[df_copy.level == 0, "level"] = "Test"
                print(metric)
                print(df_copy.columns.tolist())

                split_corr = " ".join(corr.split("_"))
                ss = 2048 if test_set_size == 5000 else test_set_size
                metric_name = metric.capitalize() if metric == "accuracy" else metric.upper()
                shift_level_box_plot(ax, df_copy, metric_name, METHODS_TO_COLORS, f"{split_corr}, Test Set Size: {ss}", hue_order=MODEL_NAMES_INDIVIDUAL, legend_loc="upper left")

                fig.tight_layout()
                # fig.savefig(os.path.join(outpath, f"{corr}-{metric}-{ss}.png"))
                fig.savefig(os.path.join(outpath, f"{corr}-{metric}-{ss}.pdf"))
                plt.close()


if __name__ == "__main__":
    # FILENAME = "mc-drop-ablation-test-results.csv"
    FILENAME = "corrupt-test-results.csv"
    # FILENAME = "test-results.csv"
    df, models = get_dataframes(BASE_RESULTS_PATH, filename=FILENAME)

    # modelnet_mc_drop_ablation(df, models)
    # print_modelnet_table_values(df, models)
    modelnet_boxplot_all_corrs(df, [m for m in models if ("universal-DeepSets" not in m and "universal-MBC" not in m)])
    modelnet_boxplot_individual(df, models)
    # print_modelnet_c_table_values(df, models)
