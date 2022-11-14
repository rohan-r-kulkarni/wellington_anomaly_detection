from class_multisimulation import MultiSimulation
from class_simulationhelper import SimulationHelpers
import numpy as np
import pandas as pd

sim = MultiSimulation()
helper = SimulationHelpers()

def gen_data(random_seed) -> pd.DataFrame:
    # hyperparameters to randomize:
    # data length, covariance matrix, number of dimensions, number of outliers. 

    n = (np.random.choice(50) + 1)*100
    dim = np.random.choice(np.arange(5, 16, 1))
    cov_mat = helper.gen_rand_cov_mat(dim, random_seed=random_seed)
    n_outlier = np.random.choice(15) + 1

    data = sim.correlated_brownian_processes_with_CO(
        n = n, 
        cov_mat = cov_mat,
        how = "full_random", 
        ma_window = 30,
        random_seed = random_seed,
        count = n_outlier,
        S0=1,
        mu=0
    )

    data = np.array([d[~np.isnan(d)] for d in data])
    data = np.array([helper.standard_scaler(d) for d in data])

    perturbed_data = np.array([
        sim.add_seasonality(
            d, 
            how = "full_random", 
            contamination=0,
        ) 
        for d in data
    ])

    perturbed_data = np.array([
        sim.add_shift(
            d, 
            how = "full_random",
            random_seed = random_seed
        )
        for d in perturbed_data
    ])

    df = pd.DataFrame(perturbed_data.T)
    # dropna
    df.dropna(inplace=True)
    return df

if __name__ == "__main__":
    data = gen_data(random_seed=42)
    helper.plot(*data.values.T, func="diff")