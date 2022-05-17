import os

from utils import ENVLIST, Runner


def get_imgnet_runs(mode: str = "train", run: int = 0) -> ENVLIST:
    envs: ENVLIST = []
    for model, slot_type, universal in [
        ("set-xformer", "random", True),
        ("set-xformer", "random", False),
        ("mbc", "random", True),
        ("mbc", "random", False),
        ("deepsets", "random", True),
        ("deepsets", "random", False),
        ("diff-em", "random", True),
        ("diff-em", "random", False)
    ]:
        for n_parallel in [4]:
            env = os.environ.copy()
            env["MODE"] = mode
            env["CLUSTERS"] = str(4)
            env["RUN"] = str(run)
            env["SLOT_DROP"] = str(0.5)
            env["UNIVERSAL"] = str(universal)
            env["N_PARALLEL"] = str(n_parallel)
            env["UNIVERSAL_K"] = str(256 // n_parallel)
            env["HEADS"] = str(4)
            env["MODEL"] = model
            env["SLOT_TYPE"] = slot_type

            envs.append(env)
    return envs


def do_imgnet_runs() -> None:  # type: ignore
    GPUS = [4, 5, 6, 7]
    # MODE = "no-mc-test"
    # MODE = "train"
    MODE = "train"

    envs = []
    for run in range(5):
        envs.extend([*get_imgnet_runs(mode=MODE, run=run)])

    print(f"there are {len(envs)} total runs")
    runner = Runner("./run.sh", GPUS, envs, test_after_train=True)
    runner.run()


def do_oracle_runs() -> None:
    envs: ENVLIST = []
    for i in range(5):
        env = os.environ.copy()
        env["MODE"] = "train"
        env["CLUSTERS"] = str(4)
        env["RUN"] = str(i)
        env["SLOT_DROP"] = str(0.5)
        env["UNIVERSAL"] = str(False)
        env["N_PARALLEL"] = str(4)
        env["UNIVERSAL_K"] = str(64)
        env["HEADS"] = str(4)
        env["MODEL"] = "oracle"
        env["SLOT_TYPE"] = "random"

        envs.append(env)

    GPUS = [4]
    print(f"there are {len(envs)} total oracle runs")
    runner = Runner("./run.sh", GPUS, envs, test_after_train=False)
    runner.run()


if __name__ == "__main__":
    do_imgnet_runs()
    # do_oracle_runs()
