import os

from utils import ENVLIST, Runner


def get_modelnet_runs(mode: str = "train", run: int = 0) -> ENVLIST:
    envs: ENVLIST = []

    models = [
        ("set-xformer", True),
        ("set-xformer", False),
        ("diff-em", True),
        ("diff-em", False),
        ("deepsets", True),
        ("deepsets", False),
        ("mbc", True),
        ("mbc", False),
    ]

    for n_parallel, universal_k in [(4, 32)]:
        for model, universal in models:
            env = os.environ.copy()
            env["SET_SIZE"] = str(1000)
            env["RUN"] = str(run)
            env["SLOT_DROP"] = str(0.5)  # comment out for the no-drop experiments
            # env["SLOT_DROP"] = str(0.0)  # comment out for drop experiments
            env["MODE"] = mode
            env["H_DIM"] = str(256)
            env["UNIVERSAL"] = str(universal)
            env["HEADS"] = str(4)
            env["N_PARALLEL"] = str(n_parallel)
            env["UNIVERSAL_K"] = str(universal_k)
            env["MODEL"] = model
            env["SLOT_TYPE"] = "random"
            env["LR"] = str(1e-3)

            envs.append(env)
    return envs


def do_modelnet_runs() -> None:  # type: ignore
    GPUS = [0, 1, 2, 3, 4, 5, 6, 7]

    envs = []
    for run in range(5):
        envs.extend([*get_modelnet_runs(mode="train", run=run)])

    print(f"there are {len(envs)} total runs")
    runner = Runner("./run.sh", GPUS, envs, test_after_train=True)
    runner.run()


def do_no_mc_drop_test_runs() -> None:  # type: ignore
    GPUS = [0, 1, 2, 3, 4, 5, 6, 7]

    envs = []
    for run in range(5):
        envs.extend([*get_modelnet_runs(mode="mc-drop-ablation-test", run=run)])

    print(f"no MC drop test: there are {len(envs)} total runs")
    runner = Runner("./run.sh", GPUS, envs)
    runner.run()


def do_modelnet_corrupt_test() -> None:  # type: ignore
    GPUS = [1, 2, 3, 4, 5, 6, 7]

    envs = []
    for run in range(5):
        envs.extend([*get_modelnet_runs(mode="corrupt-test", run=run)])

    print(f"corrupt test: there are {len(envs)} total runs")
    runner = Runner("./run.sh", GPUS, envs)
    runner.run()


if __name__ == "__main__":
    do_modelnet_runs()
    # do_no_mc_drop_test_runs()
    # do_modelnet_corrupt_test()
