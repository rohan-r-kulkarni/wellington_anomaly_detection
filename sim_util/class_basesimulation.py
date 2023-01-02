"""
IEOR4742 - Wellington Management Anomaly Detection, Fall 2022

contains base simulation class `BaseSimulation`, parent class of `Multisimulation` in `class_multisimulation`.
Part of simulation toolset.
"""

import numpy as np
import pandas as pd
import warnings
from class_simulationhelper import SimulationHelpers
from collections.abc import Iterable
from scipy.stats import norm, t
from scipy.ndimage import shift as scipyshift


class BaseSimulation:
    """The parent class of all simulation tools for Columbia-Wellington OD project.
    This class is to be inherited by class `MultiSimulation`. The primary goal to
    construct these classes was to discover and validate model options.
    """

    def __overlay(self, process: np.ndarray, super_process: Iterable[float]):
        """Private function to overlay a process on another process.
        Will check if length match before overlaying.
        Future improvement: make sure arguments are broadcast-able.
        For now, make sure one of the processes is a np.ndarray.

        :param process: First process.
        :type process: np.ndarray
        :param super_process: Second process.
        :type super_process: Iterable[float]
        :rtype: Iterable[float]
        """
        if len(process) != len(super_process):
            raise RuntimeError("Dimension mismatch. Cannot overlay.")
        return process + super_process

    def brownian_process(
        self, n: int, mu: float = 0.1, sigma: float = 0.01, S0: float = 1
    ):
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

    def geometric_brownian_process(
        self, n: int, mu: float = 0.1, sigma: float = 0.01, S0: float = 1
    ):
        """Generate a geometric brownian motion process.

        :param n: length of series
        :type n: int
        :param mu: drift parameter
        :type mu: float
        :param sigma: volatility parameter
        :type sigma: float
        :param S0: initial position
        :type S0: float
        """
        return S0 * np.exp(self.brownian_process(n, mu, sigma))

    def add_seasonality(
        self,
        process: np.ndarray,
        start_idx: int = None,
        amp: float = None,
        freq: float = None,
        diffusion: float = None,
        how: str = "full_random",
        how_diffusion: str = None,
        contamination: float = 0,
        random_seed: int = None,
        seasonality_limit: int = None,
    ):
        """Adds seasonality to the passed time series.

        :param process: Target time series
        :type process: np.ndarray
        :param start_idx: the start index of tthe added seasonality.
        :type start_idx: int
        :param amp: amplitude parameter. Higher amp induces higher oscillation.
        :type amp: float
        :param freq: frequency parameter. Higher freq induces faster oscillation.
        :type freq: float
        :param diffusion: the diffusion multiplier. Higher diffusion implies faster diffusion.
            Default to be 0.1.
        :type diffusion: float, optional.
        :param how: the random configuration of this function.
            Only ["full_random", "manual"] are implemented.
            - 'manual': do not randomize, in which case all parameters need to be passed.
            - 'full_random': randomize amp, freq, diffusion, how_diffusion
        :type how: str
        :param how_diffusion: flag to specify how does the amplitude change over time.
            only ['linear', 'sqrt', 'no'] are implemented.
            - 'linear': amplitude grows linearly over time.
            - 'sqrt': amplitude grows proportionally to the square root of time.
            - 'no': no diffusion, constant.
            Default to be 'no'.
        :type how_diffusion: str, optional.
        :param contamination: the volatility of the random contamination introdced
            to the seaonality.
        :type contamination: float
        :param random_seed: random seed, default to `None`.
        :type random_seed: int
        :param seaonality_limit: the cutoff length for added seasonality
        :type seasonality_limit: int
        :rtype: np.ndarray
        """
        if how not in ["full_random", "random_mag", "manual"]:
            warnings.warn(
                "Invalid specification for arg 'how', default to full_random."
            )
            how = "full_random"

        if random_seed:
            np.random.seed(random_seed)
        if "random" in how:
            if how == "full_random":
                seasonality_limit = np.random.choice(
                    list(range(len(process))),
                )
                # if full_random, generate amp, freq, diffusion, contamination randomly
                # using process mean and std as benchmark
                std = np.std(process)
                amp = (
                    np.random.random() / 2 * std * np.random.choice([-1, 1])
                    if not amp
                    else amp
                )
                freq = (
                    1 / len(process) * np.random.randint(10,
                                                         100) if not freq else freq
                )
                diffusion = (
                    np.random.random() * std / len(process) *
                    np.random.choice([-1, 1])
                    if not diffusion
                    else diffusion
                )
                how_diffusion = (
                    np.random.choice(["no", "linear", "sqrt"])
                    if not how_diffusion
                    else how_diffusion
                )

            elif how == "random_mag":
                if not start_idx:
                    raise RuntimeError(
                        "Specified random_mag overlay but no start_idx is provided."
                    )

        else:
            raise NotImplementedError(f"how = {how} not implemented.")

        helper = SimulationHelpers()
        seasonality = helper.gen_seasonality(
            n=len(process),
            amp=amp,
            freq=freq,
            contamination=contamination,
            how_diffusion=how_diffusion,
            diffusion=diffusion,
        )

        if seasonality_limit is not None and isinstance(seasonality_limit, int):
            seasonality[seasonality_limit:] = 0

        return self.__overlay(process, seasonality)

    def get_random_z_above_thresh(self, thresh: float):
        """Generates random standard normal statistic above given threshold

        :param thresh: threshold
        :type thresh: float
        :rtype: float
        """
        sign = thresh > 0 - (thresh <= 0)
        abs_thresh = np.abs(thresh)
        p = norm.cdf(abs_thresh)
        norm_mult = norm.cdf(np.abs(np.random.normal())) - 0.5
        z = norm.ppf(p + norm_mult * (1 - p))
        return z * sign

    def get_random_t_above_thresh(self, thresh: float, df: int):
        """Generates random t statistic above given threshold

        :param thresh: threshold
        :type thresh: float
        :param df: degrees of freedom of the t distribution.
        :type df: int
        :rtype: float
        """
        distr = t(df)
        sign = thresh > 0 - (thresh <= 0)
        abs_thresh = np.abs(thresh)
        p = distr.cdf(abs_thresh)
        mult = distr.cdf(np.abs(np.random.standard_t(df))) - 0.5
        z = distr.ppf(p + mult * (1 - p))
        return z * sign

    def add_outlier(
        self,
        process: np.ndarray,
        thresh: float = norm.ppf(0.95),
        # thresh_z = 10,
        how: str = "full_random",
        ma_window: int = 10,
        random_seed: int = None,
        count: int = 1,
        super_process=None,
        outlier_indices: Iterable[int] = None,
    ):
        """Adds outliers to a time series.

        :param process: Target time series
        :type process: np.ndarray
        :param thresh: outlier threshold, used to generate random outlier multiplier.
        :type thresh: float
        :param how: the random configuration of this function.
            Only ["full_random", "random_mag", "manual"] are implemented.
            - 'manual': do not randomize, in which case a `super_process` array needs to be passed.
            - 'random_mag': only randomize the outlier values. Must pass outlier_indices.
            - 'full_random': randomize outlier values, positions. Must pass count.
        :type how: str
        :param ma_window: the size of the moving window to compute outlier values.
        :type ma_window: int
        :param random_seed: random seed, default to `None`.
        :type random_seed: int
        :param count: number of outliers to add. Must pass in the case of how='full_random'
        :type count: int
        :param super_process: the outlier array to be overlayed on the process,
            if how='manual' is configured. Default to `None`.
        :type super_process: np.ndarray
        :param outlier_indices: the indices on which outliers are to be added.
            Must be passed if how='random_mag' is configured. Default to `None`.
        :type outlier_indices: Iterable[int]
        :rtype: np.ndarray
        """
        if how not in ["full_random", "random_mag", "manual"]:
            warnings.warn(
                "Invalid specification for arg 'how', default to full_random."
            )
            how = "full_random"

        super_process = np.zeros_like(process)
        outlier_z = np.zeros_like(process)
        if random_seed:
            np.random.seed(random_seed)
        if "random" in how:
            if how == "full_random":
                outlier_indices = np.random.choice(
                    list(range(ma_window - 1, len(process))), size=count
                )
            elif how == "random_mag":
                if outlier_indices is None:
                    raise RuntimeError(
                        "Specified random_mag overlay but no outlier_indices is provided."
                    )
                for idx in outlier_indices:
                    if idx not in range(ma_window - 1, len(process)):
                        raise ValueError(
                            f"Specified index {idx} out of valid range.")
                warnings.warn(
                    (
                        "Specified random_mag with fixed indices"
                        f"{outlier_indices}. Argument 'count' overridden."
                    )
                )
                count = len(outlier_indices)
            else:
                raise NotImplementedError(f"how = {how} not implemented.")

            global actual
            actual = outlier_indices
            print(
                f"outlier added at indices {', '.join([str(idx) for idx in outlier_indices])}"
            )
            outlier_z[outlier_indices] = [
                self.get_random_t_above_thresh(thresh, ma_window)
                for i in range(count)
                # thresh_z for i in range(count)
            ]

            super_process = outlier_z * \
                pd.Series(process).rolling(ma_window).std()

            return self.__overlay(process, super_process)

        if how == "manual":
            if not super_process:
                raise RuntimeError(
                    "Specified manual overlay but no super_process is provided."
                )

            return self.__overlay(process, super_process)

    def add_regime_change(
        self,
        process: np.ndarray,
        event_index: int,
        shift: float,
        regime_limit: int = None,
        perturb: bool = False,
        perturb_func=lambda x: 0,
    ):
        """Adds regime change to a time series.

        :param process: Target time series
        :type process: np.ndarray
        :param event_index: the desired index on which the regime change is to be added.
        :type event_index: int
        :param shift: the amount of shifting for the regime change.
        :type shift: float
        :param regime_limit: the cutoff limit of the regime change.
        :type regime_limit: int
        :param perturb: whether to add random perturbation or not.
        :type perturb: bool
        :param perturb_func: the perturbation function to add onto the series (deprecated)
        :type perturb_func: object, callable
        """
        super_process = np.zeros_like(process)
        if isinstance(shift, (int, float)):
            shift = np.array([float(shift)] * (len(process) - event_index))
        elif isinstance(shift, Iterable):
            if len(shift) > len(process) - event_index:
                shift = np.array(shift[: len(process) - event_index])
            elif len(shift) < len(process) - event_index:
                shift = np.array(
                    list(shift) + [0] *
                    (len(process) - event_index - len(shift))
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

        if regime_limit:
            if event_index + regime_limit < len(super_process):
                super_process[event_index + regime_limit + 1:] = 0

        return self.__overlay(process, super_process)

    def add_shift(
        self,
        process: np.ndarray,
        shift: int = None,
        how: str = "full_random",
        random_seed: int = None,
    ):
        """Adds shift to target time series.

        :param process: Target time series
        :type process: np.ndarray
        :param shift: the number of period to shift the time series. Negative if
            shifting backward.
        :type shift: int
        :param how: the random configuration of this function.
            Only ["full_random", "manual"] are implemented.
            - 'manual': do not randomize, in which case a `shift` int needs to be passed
            - 'full_random': randomize shift
        :type how: str
        :param random_seed: random seed, default to `None`.
        :type random_seed: int
        """
        if how not in ["full_random", "manual"]:
            warnings.warn(
                "Invalid specification for arg 'how', default to full_random."
            )
            how = "full_random"

        if random_seed:
            np.random.seed(random_seed)

        if how == "full_random":
            shift = np.random.choice(
                np.arange(-10, 11, 1)) if not shift else shift
        elif how == "manual":
            if not shift:
                raise RuntimeError(
                    "Specified manual overlay but no shift is provided.")
        else:
            raise NotImplementedError(f"how = {how} not implemented.")

        return scipyshift(process, shift, cval=np.nan)


if __name__ == "__main__":
    sim = BaseSimulation()
    helper = SimulationHelpers()

    def random_perturb(x: Iterable):
        return 0.0001 * (np.random.rand(len(x)) * 2 - 1) + 0.001 * (
            np.cos(x) + np.sin(x - 0.01)
        )

    def random_perturb_1(x: Iterable):
        # on geom brownian
        return 0.0001 * (np.random.rand(len(x)) * 2 - 1) + 0.001 * (
            np.cos(x) + np.sin(x - 0.01)
        )

    # sigma = 0.01
    # Sig = helper.gen_rand_cov_mat(
    #     8,
    # )
    # Sig /= 5
    # print(Sig)
    # data = sim.correlated_geometric_brownian_processes_with_CO(
    #     n=10000, mu = 0, sigma = 0.1, cov_mat=Sig, S0=100,
    #     ma_window = 30,
    #     how = "random_mag",
    #     outlier_indices = [6000]
    # )

    # helper.plot(*data, func = "ret")

    # seasonality example use case:
    # p = sim.geom_brownian_process(2000, mu=0.1, sigma=0.1)
    # q = sim.add_seasonality(p, how = "full_random")
    # helper.plot(p, q)
