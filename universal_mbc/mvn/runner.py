import os

from utils import ENVLIST, Runner, remove_dups


def mog_get_set_size_runs(mode: str = "train", run: int = 0) -> ENVLIST:
    envs: ENVLIST = []
    for set_size in [32, 64, 128, 256, 512]:
        for model, slot_type, universal in [
            ("set-xformer", "random", True),
            ("set-xformer", "random", False),
            ("mbc", "random", True),
            ("mbc", "random", False),
            ("deepsets", "random", True),
            ("deepsets", "random", False),
            ("diff-em", "random", True),
            ("diff-em", "random", False),
        ]:
            env = os.environ.copy()
            env["DATASET"] = "toy-mixture-of-gaussians"
            env["ATTN_ACT"] = "softmax"
            env["SET_SIZE"] = str(set_size)
            env["MODE"] = mode
            env["RUN"] = str(run)
            env["MODEL"] = model
            env["SLOT_TYPE"] = slot_type
            env["UNIVERSAL"] = str(universal)
            env["HEADS"] = str(4)
            env["FIXED"] = str(False)
            env["SLOT_DROP"] = str(0.0)
            env["SLOT_RESIDUAL"] = str(True)
            env["LN_AFTER"] = str(True)
            env["UNIVERSAL_K"] = str(128)
            env["N_PARALLEL"] = str(1)

            envs.append(env)

    print(f"set size runs: {len(envs)}")
    return envs


def mog_get_ln_after_runs(mode: str = "train", run: int = 0) -> ENVLIST:
    envs: ENVLIST = []
    for ln_after in [False, True]:
        for model, slot_type, universal in [("set-xformer", "random", True)]:
            env = os.environ.copy()
            env["DATASET"] = "toy-mixture-of-gaussians"
            env["ATTN_ACT"] = "softmax"
            env["SET_SIZE"] = str(512)
            env["MODE"] = mode
            env["RUN"] = str(run)
            env["MODEL"] = model
            env["SLOT_TYPE"] = slot_type
            env["UNIVERSAL"] = str(universal)
            env["HEADS"] = str(4)
            env["FIXED"] = str(False)
            env["SLOT_DROP"] = str(0.0)
            env["SLOT_RESIDUAL"] = str(True)
            env["LN_AFTER"] = str(ln_after)
            env["UNIVERSAL_K"] = str(128)
            env["N_PARALLEL"] = str(1)

            envs.append(env)

    print(f"ln after runs: {len(envs)}")
    return envs


def mog_get_attn_act_runs(mode: str = "train", run: int = 0) -> ENVLIST:
    envs: ENVLIST = []
    for model in ["set-xformer"]:
        for attn_act in ["slot-sigmoid", "sigmoid", "slot-softmax", "slot-exp", "softmax"]:
            env = os.environ.copy()
            env["DATASET"] = "toy-mixture-of-gaussians"
            env["ATTN_ACT"] = attn_act
            env["SET_SIZE"] = str(512)
            env["MODE"] = mode
            env["RUN"] = str(run)
            env["MODEL"] = model
            env["SLOT_TYPE"] = "random"
            env["UNIVERSAL"] = str(True)
            env["HEADS"] = str(4)
            env["FIXED"] = str(False)
            env["SLOT_DROP"] = str(0.0)
            env["SLOT_RESIDUAL"] = str(True)
            env["LN_AFTER"] = str(True)
            env["UNIVERSAL_K"] = str(128)
            env["N_PARALLEL"] = str(1)

            envs.append(env)
    print(f"attn act runs: {len(envs)}")
    return envs


def mog_get_slot_residual_runs(mode: str = "train", run: int = 0) -> ENVLIST:
    envs: ENVLIST = []
    for slot_residual in [False, True]:
        for model in ["set-xformer"]:
            env = os.environ.copy()
            env["DATASET"] = "toy-mixture-of-gaussians"
            env["ATTN_ACT"] = "softmax"
            env["SET_SIZE"] = str(512)
            env["MODE"] = mode
            env["RUN"] = str(run)
            env["MODEL"] = model
            env["SLOT_TYPE"] = "random"
            env["UNIVERSAL"] = str(True)
            env["HEADS"] = str(4)
            env["FIXED"] = str(False)
            env["SLOT_DROP"] = str(0.0)
            env["SLOT_RESIDUAL"] = str(slot_residual)
            env["LN_AFTER"] = str(True)
            env["UNIVERSAL_K"] = str(128)
            env["N_PARALLEL"] = str(1)

            envs.append(env)
    print(f"slot residual runs: {len(envs)}")
    return envs


def mog_get_head_runs(mode: str = "train", run: int = 0) -> ENVLIST:
    envs: ENVLIST = []
    for model in ["set-xformer"]:
        for n_parallel in [1, 2, 4, 8, 16]:
            for heads in [1, 4, 8, 16]:
                env = os.environ.copy()
                env["DATASET"] = "toy-mixture-of-gaussians"
                env["ATTN_ACT"] = "softmax"
                env["SET_SIZE"] = str(512)
                env["MODE"] = mode
                env["RUN"] = str(run)
                env["MODEL"] = model
                env["SLOT_TYPE"] = "random"
                env["UNIVERSAL"] = str(True)
                env["HEADS"] = str(heads)
                env["FIXED"] = str(False)
                env["SLOT_DROP"] = str(0.0)
                env["SLOT_RESIDUAL"] = str(True)
                env["LN_AFTER"] = str(True)
                env["UNIVERSAL_K"] = str(128)
                env["N_PARALLEL"] = str(n_parallel)

                envs.append(env)
    print(f"head runs: {len(envs)}")
    return envs


def mog_get_slot_drop_runs(mode: str = "train", run: int = 0) -> ENVLIST:
    envs: ENVLIST = []
    for model in ["set-xformer"]:
        for p in [0.0, 0.25, 0.5, 0.75]:
            for universal_k in [32, 64, 128, 256, 512, 1024]:
                env = os.environ.copy()
                env["DATASET"] = "toy-mixture-of-gaussians"
                env["ATTN_ACT"] = "softmax"
                env["SET_SIZE"] = str(512)
                env["MODE"] = mode
                env["RUN"] = str(run)
                env["MODEL"] = model
                env["SLOT_TYPE"] = "random"
                env["UNIVERSAL"] = str(True)
                env["HEADS"] = str(4)
                env["FIXED"] = str(False)
                env["SLOT_DROP"] = str(p)
                env["SLOT_RESIDUAL"] = str(True)
                env["LN_AFTER"] = str(True)
                env["UNIVERSAL_K"] = str(universal_k)
                env["N_PARALLEL"] = str(1)

                envs.append(env)
    print(f"slot drop and slot n runs: {len(envs)}")
    return envs


def mog_get_fixed_runs(mode: str = "train", run: int = 0) -> ENVLIST:
    envs: ENVLIST = []
    for model in ["set-xformer"]:
        env = os.environ.copy()
        env["DATASET"] = "toy-mixture-of-gaussians"
        env["ATTN_ACT"] = "softmax"
        env["SET_SIZE"] = str(512)
        env["MODE"] = mode
        env["RUN"] = str(run)
        env["MODEL"] = model
        env["SLOT_TYPE"] = "random"
        env["UNIVERSAL"] = str(True)
        env["HEADS"] = str(4)
        env["FIXED"] = str(True)
        env["SLOT_DROP"] = str(0.0)
        env["SLOT_RESIDUAL"] = str(True)
        env["LN_AFTER"] = str(True)
        env["UNIVERSAL_K"] = str(128)
        env["N_PARALLEL"] = str(1)

        envs.append(env)
    print(f"fixed runs: {len(envs)}")
    return envs


def mog_get_slot_type_runs(mode: str = "train", run: int = 0) -> ENVLIST:
    envs: ENVLIST = []
    for model in ["set-xformer"]:
        for slot_type in ["random", "deterministic"]:
            env = os.environ.copy()
            env["DATASET"] = "toy-mixture-of-gaussians"
            env["ATTN_ACT"] = "softmax"
            env["SET_SIZE"] = str(512)
            env["MODE"] = mode
            env["RUN"] = str(run)
            env["MODEL"] = model
            env["SLOT_TYPE"] = slot_type
            env["UNIVERSAL"] = str(True)
            env["HEADS"] = str(4)
            env["FIXED"] = str(False)
            env["SLOT_DROP"] = str(0.0)
            env["SLOT_RESIDUAL"] = str(True)
            env["LN_AFTER"] = str(True)
            env["UNIVERSAL_K"] = str(128)
            env["N_PARALLEL"] = str(1)

            envs.append(env)
    print(f"fixed runs: {len(envs)}")
    return envs


def remove_duplicates(envs: ENVLIST) -> ENVLIST:
    cont = True
    while cont:
        envs_new = remove_dups(envs)
        if len(envs_new) == len(envs):
            cont = False
        envs = envs_new
    return envs


def do_mc_test_runs() -> None:  # type: ignore
    def get(mode: str = "mc-ablation-test", run: int = 0) -> ENVLIST:
        """
        most of the magic of this test is in the trainer file which changes the settings on
        the fly and runs and saves results for each setting. So we just have to care about
        running this test name 5 times for the 5 runs.
        """
        envs: ENVLIST = []
        for model in ["set-xformer"]:
            for p in [0.5]:
                for universal_k in [128]:
                    env = os.environ.copy()
                    env["DATASET"] = "toy-mixture-of-gaussians"
                    env["ATTN_ACT"] = "softmax"
                    env["SET_SIZE"] = str(512)
                    env["MODE"] = mode
                    env["RUN"] = str(run)
                    env["MODEL"] = model
                    env["SLOT_TYPE"] = "random"
                    env["UNIVERSAL"] = str(True)
                    env["HEADS"] = str(4)
                    env["FIXED"] = str(False)
                    env["SLOT_DROP"] = str(p)
                    env["SLOT_RESIDUAL"] = str(True)
                    env["LN_AFTER"] = str(True)
                    env["UNIVERSAL_K"] = str(universal_k)
                    env["N_PARALLEL"] = str(1)

                    envs.append(env)
        print(f"slot drop and slot n runs: {len(envs)}")
        return envs

    GPUS = [0, 1, 2, 3, 4, 5, 6, 7]

    envs = []
    for run in range(5):
        envs.extend([*get(run=run)])

    envs = remove_duplicates(envs)
    print(f"{len(envs)=} total runs")
    runner = Runner("./run.sh", GPUS, envs, test_after_train=False)
    runner.run()


def do_motivation_run() -> None:  # type: ignore
    def get(mode: str = "mbc-motivation-example", run: int = 0) -> ENVLIST:
        """
        most of the magic of this test is in the trainer file which changes the settings on
        the fly and runs and saves results for each setting. So we just have to care about
        running this test name 5 times for the 5 runs.
        """
        envs: ENVLIST = []
        for model in ["set-xformer"]:
            for p in [0.0]:
                for universal_k in [128]:
                    for universal in [True, False]:
                        env = os.environ.copy()
                        env["DATASET"] = "toy-mixture-of-gaussians"
                        env["ATTN_ACT"] = "softmax"
                        env["SET_SIZE"] = str(512)
                        env["MODE"] = mode
                        env["RUN"] = str(run)
                        env["MODEL"] = model
                        env["SLOT_TYPE"] = "random"
                        env["UNIVERSAL"] = str(universal)
                        env["HEADS"] = str(4)
                        env["FIXED"] = str(False)
                        env["SLOT_DROP"] = str(p)
                        env["SLOT_RESIDUAL"] = str(True)
                        env["LN_AFTER"] = str(True)
                        env["UNIVERSAL_K"] = str(universal_k)
                        env["N_PARALLEL"] = str(1)

                        envs.append(env)
        print(f"slot drop and slot n runs: {len(envs)}")
        return envs

    envs = get(run=0)  # we only need one run for this experiment
    print(f"{len(envs)=} total runs")
    runner = Runner("./run.sh", [1], envs, test_after_train=False)
    runner.run()


def do_mog_runs() -> None:  # type: ignore
    GPUS = [0, 1, 2, 3, 4, 5, 6, 7] * 3
    MODE = "train"

    envs = []
    for run in range(5):
        envs.extend([
            *mog_get_head_runs(mode=MODE, run=run),
            *mog_get_slot_drop_runs(mode=MODE, run=run),
            *mog_get_ln_after_runs(mode=MODE, run=run),
            *mog_get_slot_residual_runs(mode=MODE, run=run),
            *mog_get_attn_act_runs(mode=MODE, run=run),
            *mog_get_set_size_runs(mode=MODE, run=run),
            *mog_get_slot_type_runs(mode=MODE, run=run),
            *mog_get_fixed_runs(mode=MODE, run=run),
        ])

    envs = remove_duplicates(envs)
    print(f"{len(envs)=} total runs")
    runner = Runner("./run.sh", GPUS, envs, test_after_train=False)
    runner.run()


if __name__ == "__main__":
    do_mog_runs()
    # do_mc_test_runs()
    # do_motivation_run()
