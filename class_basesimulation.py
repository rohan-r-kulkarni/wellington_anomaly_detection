from matplotlib.pyplot import axes
import numpy as np
import pandas as pd
from collections.abc import Iterable
import warnings
from scipy.stats import norm


class BaseSimulation:
    def unit_root(self, n: int, mu: float = 0, sigma: float = 1):
        return (sigma * (2 * np.random.rand(n) - 1) + mu).cumsum(axis=0)

    def brownian_process(self, n: int, mu=0.1, sigma=0.01):
        dt = 1 / n
        t = np.linspace(0, 1, n)
        W = np.random.standard_normal(size=n)
        W = np.cumsum(W) * np.sqrt(dt)  ### standard brownian motion ###
        X = (mu - 0.5 * sigma**2) * t + sigma * W
        return X

    def geom_brownian_process(self, n: int, mu=0.1, sigma=0.01, S0=1):
        return S0 * np.exp(self.brownian_process(n, mu, sigma))

    def correlated_process(self, process_func, corr: float, **kwargs):
        z1 = process_func(**kwargs)
        z2 = process_func(**kwargs)
        return z1, corr * z1 + np.sqrt(1 - corr**2) * z2

    def correlated_unit_root(
        self, n: int, corr: float, mu: float = 0, sigma: float = 1
    ):
        return self.correlated_process(
            process_func=self.unit_root, 
            corr=corr,
            kwargs={'n':n, 'mu':mu, 'sigma':sigma}
        )

    def correlated_brownian_process(self, corr: float, n: int, mu=0.1, sigma=0.01):
        return self.corrleated_process(
            process_func=self.brownian_process, 
            corr=corr, 
            kwargs={'n':n, 'mu':mu, 'sigma':sigma}
        )

    def correlated_geometric_brownian_process(
        self, corr: float, n: int, mu: float = 0.1, sigma: float = 0.01
    ):
        z1, z2 = self.correlated_brownian_process(
            corr,
            n,
            mu, 
            sigma
        )
        return np.exp(z1), np.exp(z2)

    def __overlay(self, process, super_process):
        if len(process) != len(super_process):
            raise RuntimeError("Dimension mismatch. Cannot overlay.")
        return process + super_process

    def add_seasonality(
        self,
        process,
        start,
        amp,
    ):

        # TODO
        pass

    def get_random_z_above_thresh(self, thresh: float):
        sign = thresh > 0 - (thresh <= 0)
        abs_thresh = np.abs(thresh)
        p = norm.cdf(abs_thresh)
        norm_mult = norm.cdf(np.abs(np.random.normal())) - 0.5
        z = norm.ppf(p + norm_mult * (1 - p))
        return z * sign

    def add_outlier(
        self,
        process,
        thresh_z=norm.ppf(0.95),
        how="random",
        ma_window=10,
        random_seed=None,
        count=1,
    ):
        if how not in ["random", "manual"]:
            warnings.warn("Invalid specification for arg 'how', default to random.")
            how = "random"

        super_process = np.zeros_like(process)
        outlier_z = np.zeros_like(process)
        if random_seed:
            np.random.seed(random_seed)

        if how == "random":
            outlier_indices = np.random.choice(
                list(range(ma_window - 1, len(process))), size=count
            )
            print(f"outlier added at indices {', '.join([str(idx) for idx in outlier_indices])}")
            outlier_z[outlier_indices] = [
                self.get_random_z_above_thresh(thresh_z) for i in range(count)
            ]
            # TODO: modify me to actually work in numpy. This is essentially pseudo-code.
            super_process = outlier_z * pd.Series(process).rolling(ma_window).std()
            return self.__overlay(process, super_process)

    def add_regime_change(
        self, process, event_index, shift, perturb=False, perturb_func=lambda x: 0
    ):
        super_process = np.zeros_like(process)
        if isinstance(shift, (int, float)):
            shift = np.array([float(shift)] * (len(process) - event_index))
        elif isinstance(shift, Iterable):
            if len(shift) > len(process) - event_index:
                shift = np.array(shift[: len(process) - event_index])
            elif len(shift) < len(process) - event_index:
                shift = np.array(
                    list(shift) + [0] * (len(process) - event_index - len(shift))
                )
        else:
            raise ValueError(
                (
                    "Unexpected shift type. Shift can either be an array or a number. "
                    f"type(shift) = {type(shift)}."
                )
            )
        if perturb:
            shift += perturb_func(shift)
        super_process[event_index:] = shift

        return self.__overlay(process, super_process)


sim = BaseSimulation()
# df = pd.DataFrame()
# df["1"], df["2"] = sim.correlated_unit_root(10, 0.1)
# pd.Series(sim.unit_root(1000, mu = 0, sigma = 20)).plot()
# p = 100 * sim.unit_root(1000, mu = 0, sigma = 0.1)


def random_perturb(x: Iterable):
    return 0.0001 * (np.random.rand(len(x)) * 2 - 1) + 0.001 * (
        np.cos(x) + np.sin(x - 0.01)
    )


def random_perturb_1(x: Iterable):
    # on geom brownian
    return 0.0001 * (np.random.rand(len(x)) * 2 - 1) + 0.001 * (
        np.cos(x) + np.sin(x - 0.01)
    )


p = sim.geom_brownian_process(1000, mu=0.1, sigma=1)
# q = pd.Series(
#     sim.add_regime_change(p, 900, 0.5, perturb=True, perturb_func=random_perturb)
# )
q = pd.Series(
    sim.add_outlier(p, count = 2, thresh_z = 3)
)

p = pd.Series(p)
import matplotlib.pyplot as plt


def plot(*args, figsize=(10, 7), func=None):
    fig, ax = plt.subplots(1, len(args), sharey=True)
    for i, arg in enumerate(args):
        if func == "log":
            np.log(arg).plot(grid=True, figsize=figsize, ax=ax[i])
        elif func == "ret":
            arg.pct_change().plot(grid=True, figsize=figsize, ax=ax[i])
        else:
            arg.plot(grid=True, figsize=figsize, ax=ax[i])


# fig, ax = plt.subplots(1,2, sharey=True)
# np.log(p).plot(grid=True, figsize=(10, 7), ax = ax[0])
# np.log(q).plot(grid=True, figsize=(10, 7), ax = ax[1])

plot(p, q)
plot(p, q, func="ret")
