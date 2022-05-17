import os.path
from typing import List

import pandas as pd  # type: ignore
from universal_mbc.plot import Q, df_query, get_dataframes

j = os.path.join
BASE_RESULTS_PATH = j("results", "ImageNetClusters")

outpath = j(BASE_RESULTS_PATH, "plots")
os.makedirs(outpath, exist_ok=True)


def get_sse_kwargs(variant: str = "UNIVERSAL", k: int = 64) -> Q:
    prefix = {"UNIVERSAL": "sse.0", "MBC": "encoder.0", "HIERARCHICAL": "sse.0"}[variant]
    return {
        ":ln_after": True,
        f"{prefix}:attn_act": "softmax",
        f"{prefix}.slots:slot_type": "random",
        f"{prefix}:heads": 4,
        f"{prefix}:slot_drop": 0.5,
        f"{prefix}:slot_residual": True,
        f"{prefix}.slots:K": k,
        f"{prefix}.slots:fixed": False
    }


DEFAULT_ST_KWARGS: Q = {}
DEFAULT_DS_KWARGS: Q = {}
DEFAULT_DIFFEM_KWARGS: Q = {}


def get_default_kwargs(model: str) -> Q:
    if "universal" in model:  # this case must go first
        return {**get_sse_kwargs("UNIVERSAL"), ":n_parallel": 4}
    elif "Hierarchical" in model:
        return get_sse_kwargs("HIERARCHICAL")
    elif "Transformer" in model or "Xformer" in model:
        return DEFAULT_ST_KWARGS
    elif "DiffEM" in model:
        return DEFAULT_DIFFEM_KWARGS
    elif "DeepSets" in model:
        return DEFAULT_DS_KWARGS
    elif "Oracle" in model:
        return {}
    else:
        return {**get_sse_kwargs("MBC", k=4), "encoder.0:slot_drop": 0.0}


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    # different test runs can give different floating point errors on the results so we have to drop the results columns as well
    duplicate_cols = df.columns[:1].tolist() + df.columns[4:].tolist()
    df = df.drop_duplicates(subset=duplicate_cols)
    return df


def print_imgnet_table_values(df: pd.DataFrame, models: List[str]) -> None:
    for model in df.model.unique():
        for metric in ["loss", "adj_rand_idx"]:
            vals = []

            df_test = df[(df.model == model)]

            query = get_default_kwargs(model)
            df_test = df_query(df_test, **query)  # type: ignore
            df_test = drop_duplicates(df_test)

            df_test = df_test[metric].to_numpy()
            if metric in ["adj_rand_idx"]:
                df_test *= 100

            # print(df_test_ss)
            mu = df_test.mean()
            std = df_test.std()
            vals.append(f"{mu:.2f}$\pm${std:.2f}")

            print(f"{model=} {metric=} {' & '.join(vals)}")


if __name__ == "__main__":
    df, models = get_dataframes(BASE_RESULTS_PATH)

    print_imgnet_table_values(df, models)
