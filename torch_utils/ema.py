import numpy as np
import click
import glob


# from Karras Paper
def sigma_rel_to_gamma(sigma_rel):
    t = sigma_rel**-2
    gamma = np.roots([1, 7, 16 - t, 12 - t]).real.max()
    return gamma


# from Karras Paper
def p_dot_p(t_a, gamma_a, t_b, gamma_b):
    t_ratio = t_a / t_b
    t_exp = np.where(t_a < t_b, gamma_b, -gamma_a)
    t_max = np.maximum(t_a, t_b)
    num = (gamma_a + 1) * (gamma_b + 1) * t_ratio**t_exp
    den = (gamma_a + gamma_b + 1) * t_max
    return num / den


# from Karras Paper
def solve_weights(t_i, gamma_i, t_r, gamma_r):
    def rv(x):
        return x.astype(np.float64).reshape(-1, 1)

    def cv(x):
        return x.astype(np.float64).reshape(1, -1)

    A = p_dot_p(rv(t_i), rv(gamma_i), cv(t_i), cv(gamma_i))
    B = p_dot_p(rv(t_i), rv(gamma_i), cv(t_r), cv(gamma_r))
    X = np.linalg.solve(A, B)
    return X


# ----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of floats.
# Example: '1.0,2.0,3.0' returns [1.0, 2.0, 3.0]


def parse_float_list(s):
    if isinstance(s, list):
        return s
    return [float(x) for x in s.split(",")]


# ----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5' returns [1, 2, 5]


def parse_int_list(s):
    if isinstance(s, list):
        return s
    return [int(x) for x in s.split(",")]


@click.command()
@click.option(
    "--ema_sigmas", type=parse_float_list, default="0.05,0.10", help="EMA sigmas"
)
@click.option("--snapshot_dir", type=str, required=True, help="Snapshot directory")
@click.option(
    "--target_sigmas",
    type=parse_float_list,
    default="0.075,",
    help="Target sigmas",
)
@click.option(
    "--target_steps", type=parse_int_list, default="1000,", help="Target steps"
)
def main(ema_sigmas, snapshot_dir, target_sigmas, target_steps):
    import pickle
    import copy

    # get all snapshots
    snapshots = glob.glob(f"{snapshot_dir}/network-snapshot-*.pkl")
    pickles = [pickle.load(open(s, "rb")) for s in snapshots]
    steps = [p["step"] for p in pickles]
    t_i = np.concatenate([np.array(steps) for _ in ema_sigmas])
    gamma_i = np.concatenate(
        [np.ones(len(steps)) * sigma_rel_to_gamma(s) for s in ema_sigmas]
    )
    t_r = np.concatenate([np.array(target_steps) for _ in target_sigmas])
    gamma_r = np.concatenate(
        [np.ones(len(target_steps)) * sigma_rel_to_gamma(s) for s in target_sigmas]
    )
    weights = solve_weights(t_i, gamma_i, t_r, gamma_r)
    models = sum([[p[f"ema_{s}"] for p in pickles] for s in ema_sigmas], [])
    for target_step in range(len(target_steps)):
        data = {}
        for target_sigma in range(len(target_sigmas)):
            weight_column = weights[:, target_sigma * len(target_steps) + target_step]
            target_model = copy.deepcopy(models[0])
            for name, param in target_model.named_parameters():
                param.data = sum(
                    [
                        weight * model.state_dict()[name]
                        for weight, model in zip(weight_column, models)
                    ]
                )
            data[f"ema_{target_sigmas[target_sigma]}"] = target_model
        with open(
            f"{snapshot_dir}/posthoc-{target_steps[target_step]}.pkl",
            "wb",
        ) as f:
            pickle.dump(data, f)


if __name__ == "__main__":
    main()
