"""
IEOR4742 - Wellington Management Anomaly Detection, Fall 2022

contains helper functions used to implement previous classes. It also contains other helper method for plotting.
Part of simulation toolset.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections.abc import Iterable
from matplotlib.pyplot import cm
from typing import Iterable


class SimulationHelpers:
    """The helper class to store helper functions for data simulation.
    """

    def __prepare_ax(self, i: int, n: int, row_lim: int, ax: plt.Axes):
        """
        Private function to aid graphing. Only called in self.plot().

        :param i: index of ax
        :type i: int
        :param n: total number of subplots in the ax object
        :type n: int
        :param row_lim: the maximum number of subplots per row
        :type row_lim: int
        :param ax: the matplotlib.Axes object
        :type ax: plt.Axes
        :rtype: plt.Axes
        """
        if n == 1:
            return ax
        if n > row_lim:
            return ax[i // row_lim, i % row_lim]
        else:
            return ax[i % row_lim]

    def zip_series_for_plot(self, *args):
        """
        zipping time series that need to be plotted together.

        :param args: time series Iterable
        :type args: Iterable, optional
        """
        return [list(a) for a in zip(*args)]

    def plot(
        self,
        *args,
        outliers: Iterable = None,
        figsize: Iterable = (20, 14),
        func: str = None,
        row_lim: int = 4
    ):
        """
        Plot time series and mark out outliers.

        Format:
            plot(
                series_1, series_2,
                ...,
                outliers=[outlier_idx_1, outlier_idx_2, ...]
            )

        :param args: time series Iterable, optional
        :type args: Iterables, e.g. `pd.Series`.
        :outliers: list of outlier indices to be overlayed on top of each subplot.
            Must be of the same length as args. Default to `None`.
        :type outliers: List[Iterable[int]], optional
        :param figsize: Iterable specifying plt figsize, default to (20, 14)
        :type figsize: Iterable, conventionally tuple, optional.
        :param func: transformation applied to each series before plotting.
            only ['log', 'ret', 'diff'] are implemented.
            - 'log': equivalent to lambda x: np.log(x)
            - 'ret': equivalent to lambda x: x.pct_change() in pandas
            - 'diff': equivalent to lambda x: x.diff() in pandas
            optional, default to `None`.
        :type func: str
        :param row_lim: the maximum number of subplots per row
        :type row_lim: int
        """
        n = len(args)
        if n > row_lim:
            _, ax = plt.subplots((n - 1) // row_lim + 1, row_lim, sharey=True)
        else:
            _, ax = plt.subplots(1, min(row_lim, n), sharey=True)

        for i, arg in enumerate(args):
            if isinstance(arg, list):
                arg = [pd.Series(a) for a in arg]
            else:
                arg = [pd.Series(arg)]

            if func == "log":
                for a in arg:
                    np.log(a).plot(
                        grid=True,
                        figsize=figsize,
                        ax=self.__prepare_ax(i, n, row_lim, ax),
                    )
            elif func == "ret":
                for a in arg:
                    a.pct_change().plot(
                        grid=True,
                        figsize=figsize,
                        ax=self.__prepare_ax(i, n, row_lim, ax),
                    )
            elif func == "diff":
                for a in arg:
                    a.diff().plot(
                        grid=True,
                        figsize=figsize,
                        ax=self.__prepare_ax(i, n, row_lim, ax),
                    )
            else:
                for a in arg:
                    a.plot(
                        grid=True,
                        figsize=figsize,
                        ax=self.__prepare_ax(i, n, row_lim, ax),
                    )

        if outliers is not None:
            if not isinstance(outliers, Iterable) or len(outliers) != len(args):
                raise RuntimeError(
                    "Invalid outlier passed. Must have the same length as *args."
                )

            for i, l in enumerate(outliers):
                cmap = iter(cm.rainbow(np.linspace(0, 1, len(l))))
                for point in l:
                    pos = (
                        ax[i // row_lim, i % row_lim]
                        if n > row_lim
                        else ax[i % row_lim]
                    )
                    pos.axvline(x=point, color=next(cmap),
                                linestyle="--", alpha=0.8)
        return ax

    def gen_rand_cov_mat(self, n: int, random_seed: int = None):
        """Generates a positive definite random covariance matrix.

        :param n: number of rows/columns of the matrix
        :type n: int
        :param random_seed: random seed, default to `None`.
        :type random_seed: int
        :rtype: np.array
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        A = np.random.rand(n, n)
        Cov = A.dot(A.T)
        return Cov

    def corr_from_cov(self, x: np.array):
        """Convert a covariance matrix to a correlation matrix.

        :param x: Covariance matrix
        :type x: np.array
        :rtype: np.array
        """
        v = np.sqrt(np.diag(x))
        outer_v = np.outer(v, v)
        corr = x / outer_v
        corr[x == 0] = 0
        return corr

    def brownian_process(self, n: int, mu=0.1, sigma: float = 0.01, S0: float = 1):
        """Generate a brownian motion process.

        :param n: length of series
        :type n: int
        :param mu: drift parameter
        :type mu: float
        :param sigma: volatility parameter
        :type sigma: float
        :param S0: initial position
        :type S0: float
        """
        dt = 1 / n
        t = np.linspace(0, 1, n)
        W = np.random.standard_normal(size=n)
        W = np.cumsum(W) * np.sqrt(dt)  # standard brownian motion ###
        X = (mu - 0.5 * sigma**2) * t + sigma * W
        return X * S0

    def standard_scaler(self, process: np.ndarray):
        """Scales a series by its mean and std.

        :param process: the time series to be scaled
        :type process: np.array
        :rtype: np.ndarray
        """
        mean = np.mean(process)
        std = np.std(process)
        return (process - mean) / std

    def gen_seasonality(
        self,
        n: int,
        amp: float,
        freq: float,
        contamination: float,
        how_diffusion: str = "no",
        diffusion: float = 0.1,
    ):
        """Seasonality generation function copied from class `BaseSimulation`.

        :param n: length of series
        :type n: int
        :param amp: amplitude parameter. Higher amp induces higher oscillation.
        :type amp: float
        :param freq: frequency parameter. Higher freq induces faster oscillation.
        :type freq: float
        :param contamination: the volatility of the random contamination introdced
            to the seaonality.
        :type contamination: float
        :param how_diffusion: flag to specify how does the amplitude change over time.
            only ['linear', 'sqrt', 'no'] are implemented.
            - 'linear': amplitude grows linearly over time.
            - 'sqrt': amplitude grows proportionally to the square root of time.
            - 'no': no diffusion, constant.
            Default to be 'no'.
        :type how_diffusion: str, optional.
        :param diffusion: the diffusion multiplier. Higher diffusion implies faster diffusion.
            Default to be 0.1.
        :type diffusion: float, optional.
        :rtype: np.array
        """

        x = np.arange(0, n, 1)
        w = self.brownian_process(n, mu=0, sigma=contamination)
        if how_diffusion == "linear":
            amp = x * amp * diffusion
        elif how_diffusion == "sqrt":
            amp = np.sqrt(x) * amp * diffusion
        elif how_diffusion == "no":
            pass

        return amp * np.cos(x * freq) + w
