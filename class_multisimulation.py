"""
IEOR4742 - Wellington Management Anomaly Detection, Fall 2022

contains inherited simulation class `MultiSimulation`, child class of `BaseSimulation`
in `class_basesimulation` in `class_basesimulation`. Part of simulation toolset.
"""

import numpy as np
import warnings
from class_basesimulation import BaseSimulation
from collections.abc import Iterable
from typing import Union, Callable


class MultiSimulation(BaseSimulation):
    """This class inherits class `BaseSimulation`, on top of its parent's basic
    functionalities, this class adds the capacity of simulating multiple correlated
    processes. The primary goal to construct these classes was to discover and
    validate model options.
    """

    def correlated_process(
        self,
        process_func: Callable,
        corr: Union[float, None] = None,
        cov_mat: Union[np.ndarray, None] = None,
        **kwargs,
    ):
        """Kernel function to generate correlated processes. Although not private
        or protected, it's designed to be wrapped into other generation functions
        by specifying the process_func.

        :param process_func: the process function to generate one desired sires
        :type process_func: Callable
        :param corr: either a correlation coefficient or `None`
        :type corr: Union[float, None]
        :param cov_mat: the covariance matrix, should want to generate processes
            with more than two dimensions.
        :type cov_mat: np.ndarray, optional
        :rtype: np.ndarray
        """
        if corr and isinstance(corr, float):  # if corr is a real number
            z1 = process_func(**kwargs)
            z2 = process_func(**kwargs)
            return z1, corr * z1 + np.sqrt(1 - corr**2) * z2
        else:  # if corr is not a real number
            # check if a covariance matrix is passed
            if cov_mat is None:
                raise ValueError(
                    "Either a corr value or a cov_mat must be passed.")
            try:
                L = np.linalg.cholesky(cov_mat)
            except:
                raise ValueError("Not a valid covariance matrix.")
            n = cov_mat.shape[0]

            z = np.array([process_func(**kwargs) for _ in range(n)])

            return np.dot(L, z)

    def correlated_brownian_process(
        self,
        n: int,
        corr: Union[float, None] = None,
        mu: float = 0.1,
        sigma: float = 0.01,
        cov_mat: Union[np.ndarray, None] = None,
        S0: float = 1,
    ):
        """Generates correlated bownian motions.

        :param n: length of generated series.
        :type n: int
        :param corr: either a correlation coefficient or `None`
        :type corr: Union[float, None]
        :param mu: drift parameter of the brownian motions.
        :type mu: float
        :param sigma: volatility parameter of the brownian motions.
        :type sigma: float
        :param cov_mat: the covariance matrix, should want to generate processes
            with more than two dimensions.
        :type cov_mat: np.ndarray, optional
        :param S0: initial position of the brownian motions.
        :type S0: float, optional
        """
        return self.correlated_process(
            process_func=self.brownian_process,
            corr=corr,
            n=n,
            mu=mu,
            sigma=sigma,
            cov_mat=cov_mat,
            S0=S0,
        )

    def correlated_geometric_brownian_process(
        self,
        n: int,
        corr: Union[float, None] = None,
        mu: float = 0.1,
        sigma: float = 0.01,
        cov_mat: np.ndarray = None,
        S0: float = 1,
    ):
        """Generates correlated geometric bownian motions.

        :param n: length of generated series.
        :type n: int
        :param corr: either a correlation coefficient or `None`
        :type corr: Union[float, None]
        :param mu: drift parameter of the brownian motions.
        :type mu: float
        :param sigma: volatility parameter of the brownian motions.
        :type sigma: float
        :param cov_mat: the covariance matrix, should want to generate processes
            with more than two dimensions.
        :type cov_mat: np.ndarray, optional
        :param S0: initial position of the brownian motions.
        :type S0: float, optional
        """
        if cov_mat is None:
            z1, z2 = self.correlated_brownian_process(
                corr=corr, n=n, mu=mu, sigma=sigma, cov_mat=cov_mat, S0=1
            )
            return S0 * np.exp(z1), S0 * np.exp(z2)
        else:
            return S0 * np.exp(
                self.correlated_brownian_process(
                    corr=corr, n=n, mu=mu, sigma=sigma, cov_mat=cov_mat, S0=1
                )
            )

    def correlated_processes_with_correlated_outliers(
        self,
        process_func: Callable,
        corr: float,
        cov_mat: np.ndarray = None,
        how: str = "full_random",
        ma_window: int = 10,
        random_seed: int = None,
        count: int = 1,
        outlier_indices: Iterable[int] = None,
        **kwargs,
    ):
        """Kernel function to generate correlated processes with correlated outliers.
        Although not private protected, it's designed to be wrapped into other
        generation functions by specifying the process_func.

        :param process_func: the process function to generate one desired sires
        :type process_func: Callable
        :param corr: either a correlation coefficient or `None`
        :type corr: Union[float, None]
        :param cov_mat: the covariance matrix, should want to generate processes
            with more than two dimensions.
        :type cov_mat: np.ndarray, optional
        :param how: the random configuration of this function.
            Only ["full_random", "random_mag"] are implemented.
            - 'random_mag': randomize only the outlier values, in which case
                `outlier_indices` needs to be passed
            - 'full_random': randomize outliers values and indices
        :type how: str
        :param ma_window: the size of the moving window to compute outlier values.
        :type ma_window: int
        :param random_seed: random seed, default to `None`.
        :type random_seed: int
        :param count: number of outliers to add. Must pass in the case of how='full_random'
        :type count: int
        :param outlier_indices: the indices on which outliers are to be added.
            Must be passed if how='random_mag' is configured. Default to `None`.
        :type outlier_indices: Iterable[int]
        :param kwargs: other kwargs to be passed into the process_func Callable.
        :rtype: np.ndarray
        """
        if random_seed:
            np.random.seed(random_seed)
        # Introduce similar outliers to independent series
        # Then correlate them
        if how == "full_random":
            outlier_indices = (
                np.random.choice(
                    list(range(ma_window - 1, len(process_func(**kwargs)))), size=count
                )
                if outlier_indices is None
                else outlier_indices
            )
        elif how == "random_mag":
            if not outlier_indices:
                raise RuntimeError(
                    "Specified semi-random overlay but no outlier_indices is provided."
                )
            for idx in outlier_indices:
                if idx not in range(ma_window - 1, len(process_func(**kwargs))):
                    raise ValueError(
                        f"Specified index {idx} out of valid range.")
            warnings.warn(
                (
                    "Specified random_mag with fixed indices"
                    f"{outlier_indices}. Argument 'count' overridden."
                )
            )
            count = len(outlier_indices)

        if corr and isinstance(corr, float):  # if corr is a real number
            z1 = process_func(**kwargs)
            z2 = process_func(**kwargs)

            z1 = self.add_outlier(
                z1,
                how="random_mag",
                ma_window=ma_window,
                random_seed=random_seed,
                outlier_indices=outlier_indices,
            )
            z2 = self.add_outlier(
                z2,
                how="random_mag",
                ma_window=ma_window,
                random_seed=random_seed,
                outlier_indices=outlier_indices,
            )

            return z1, corr * z1 + np.sqrt(1 - corr**2) * z2
        else:  # if corr is not a real number
            # check if a covariance matrix is passed
            if cov_mat is None:
                raise ValueError(
                    "Either a corr value or a cov_mat must be passed.")
            try:
                L = np.linalg.cholesky(cov_mat)
            except:
                raise ValueError("Not a valid covariance matrix.")

            n = cov_mat.shape[0]
            z = [process_func(**kwargs) for _ in range(n)]
            z = [
                self.add_outlier(
                    process,
                    how="random_mag",
                    ma_window=ma_window,
                    random_seed=random_seed,
                    outlier_indices=outlier_indices,
                )
                for process in z
            ]

            return np.dot(L, z)

    def correlated_brownian_processes_with_CO(
        self,
        n: int,
        corr: Union[float, None] = None,
        mu: float = 0.1,
        sigma: float = 0.01,
        cov_mat: np.ndarray = None,
        S0: float = 1,
        how: str = "full_random",
        ma_window: int = 10,
        random_seed: int = None,
        count: int = 1,
        outlier_indices: Iterable[int] = None,
    ):
        """Generate correlated brownian motion processes with correlated outliers.

        :param n: length of generated series.
        :type n: int
        :param corr: either a correlation coefficient or `None`
        :type corr: Union[float, None]
        :param mu: drift parameter of the brownian motions.
        :type mu: float
        :param sigma: volatility parameter of the brownian motions.
        :type sigma: float
        :param cov_mat: the covariance matrix, should want to generate processes
            with more than two dimensions.
        :type cov_mat: np.ndarray, optional
        :param S0: initial position of the brownian motions.
        :type S0: float, optional
        :param how: the random configuration of this function.
            Only ["full_random", "random_mag"] are implemented.
            - 'random_mag': randomize only the outlier values, in which case
                `outlier_indices` needs to be passed
            - 'full_random': randomize outliers values and indices
        :type how: str
        :param ma_window: the size of the moving window to compute outlier values.
        :type ma_window: int
        :param random_seed: random seed, default to `None`.
        :type random_seed: int
        :param count: number of outliers to add. Must pass in the case of how='full_random'
        :type count: int
        :param outlier_indices: the indices on which outliers are to be added.
            Must be passed if how='random_mag' is configured. Default to `None`.
        :type outlier_indices: Iterable[int]
        :rtype: np.ndarray
        """
        # write this by calling the function above.
        if random_seed:
            np.random.seed(random_seed)
        if how == "full_random":
            mu = np.random.random() * S0 * np.random.choice([-1, 1])
            sigma = np.random.random() * mu / 2
        return self.correlated_processes_with_correlated_outliers(
            process_func=self.brownian_process,
            corr=corr,
            cov_mat=cov_mat,
            how=how,
            ma_window=ma_window,
            random_seed=random_seed,
            outlier_indices=outlier_indices,
            count=count,
            S0=S0,
            n=n,
            mu=mu,
            sigma=sigma,
        )

    def correlated_geometric_brownian_processes_with_CO(
        self,
        n: int,
        corr: Union[float, None] = None,
        mu: float = 0.1,
        sigma: float = 0.01,
        cov_mat: np.ndarray = None,
        S0: float = 1,
        how: str = "full_random",
        ma_window: int = 10,
        random_seed: int = None,
        count: int = 1,
        outlier_indices: Iterable[int] = None,
    ):
        """Generate correlated geometric brownian motion processes with correlated outliers.

        :param n: length of generated series.
        :type n: int
        :param corr: either a correlation coefficient or `None`
        :type corr: Union[float, None]
        :param mu: drift parameter of the brownian motions.
        :type mu: float
        :param sigma: volatility parameter of the brownian motions.
        :type sigma: float
        :param cov_mat: the covariance matrix, should want to generate processes
            with more than two dimensions.
        :type cov_mat: np.ndarray, optional
        :param S0: initial position of the brownian motions.
        :type S0: float, optional
        :param how: the random configuration of this function.
            Only ["full_random", "random_mag"] are implemented.
            - 'random_mag': randomize only the outlier values, in which case
                `outlier_indices` needs to be passed
            - 'full_random': randomize outliers values and indices
        :type how: str
        :param ma_window: the size of the moving window to compute outlier values.
        :type ma_window: int
        :param random_seed: random seed, default to `None`.
        :type random_seed: int
        :param count: number of outliers to add. Must pass in the case of how='full_random'
        :type count: int
        :param outlier_indices: the indices on which outliers are to be added.
            Must be passed if how='random_mag' is configured. Default to `None`.
        :type outlier_indices: Iterable[int]
        :rtype: np.ndarray
        """
        if random_seed:
            np.random.seed(random_seed)
        if cov_mat is None:
            z1, z2 = self.correlated_brownian_processes_with_CO(
                corr=corr,
                n=n,
                mu=mu,
                sigma=sigma,
                cov_mat=cov_mat,
                ma_window=ma_window,
                random_seed=random_seed,
                outlier_indices=outlier_indices,
                count=count,
                how=how,
                S0=1,
            )
            return S0 * np.exp(z1), S0 * np.exp(z2)
        else:
            return S0 * np.exp(
                self.correlated_brownian_processes_with_CO(
                    corr=corr,
                    n=n,
                    mu=mu,
                    sigma=sigma,
                    cov_mat=cov_mat,
                    ma_window=ma_window,
                    random_seed=random_seed,
                    outlier_indices=outlier_indices,
                    count=count,
                    how=how,
                    S0=1,
                )
            )
