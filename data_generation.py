from class_multisimulation import MultiSimulation
from class_simulationhelper import SimulationHelpers
import numpy as np
import pandas as pd

sim = MultiSimulation()
helper = SimulationHelpers()


def gen_data(
    random_seed: int, n: int = None, dim: int = None, n_outlier: int = None
) -> pd.DataFrame:
    """Use only a random seed to generate multivariate time series.

    :param random_seed: random seed.
    :type random_seed: int
    :param n: number of periods of the time series, default to be `None`.
    :type n: int, optional
    :param dim: number of dimensions of the time series, default to be `None`
    :type dim: int, optional
    :rtype: pd.DataFrame
    """
    # hyperparameters to randomize:
    # data length, covariance matrix, number of dimensions, number of outliers.
    n = (np.random.choice(50) + 1) * 100 if n is None else n
    dim = np.random.choice(np.arange(5, 16, 1)) if dim is None else dim
    # manual specification of covairance matrix is not implemented.
    # since speicifcation defeats the purpose of randomization
    cov_mat = helper.gen_rand_cov_mat(dim, random_seed=random_seed)
    n_outlier = np.random.choice(15) + 1 if n_outlier is None else n_outlier

    # generate correlated brownian proceses with correlated outliers.
    data = sim.correlated_brownian_processes_with_CO(
        n=n,
        cov_mat=cov_mat,
        how="full_random",
        ma_window=30,
        random_seed=random_seed,
        count=n_outlier,
        S0=1,
        mu=0,
    )

    # scale and dropna
    data = np.array([d[~np.isnan(d)] for d in data])
    data = np.array([helper.standard_scaler(d) for d in data])

    # add seasonalilties
    perturbed_data = np.array(
        [
            sim.add_seasonality(
                d,
                how="full_random",
                contamination=0,
            )
            for d in data
        ]
    )

    # add random shifts
    perturbed_data = np.array(
        [
            sim.add_shift(d, how="full_random", random_seed=random_seed)
            for d in perturbed_data
        ]
    )

    # for future developers: can add other random transformations should one desire

    df = pd.DataFrame(perturbed_data.T)
    # dropna
    df.dropna(inplace=True)
    return df


if __name__ == "__main__":
    data = gen_data(random_seed=42)
    helper.plot(*data.values.T, func="diff")
