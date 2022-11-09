import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections.abc import Iterable
import warnings
from scipy.stats import norm, t
from class_simulationhelper import SimulationHelpers


class BaseSimulation:
    """Write me"""
    def __overlay(self, process, super_process):
        if len(process) != len(super_process):
            raise RuntimeError("Dimension mismatch. Cannot overlay.")
        return process + super_process

    def brownian_process(self, n: int, mu=0.1, sigma=0.01, S0=1):
        dt = 1 / n
        t = np.linspace(0, 1, n)
        W = np.random.standard_normal(size=n)
        W = np.cumsum(W) * np.sqrt(dt)  ### standard brownian motion ###
        X = (mu - 0.5 * sigma**2) * t + sigma * W
        return X*S0

    def geom_brownian_process(self, n: int, mu=0.1, sigma=0.01, S0=1):
        return S0 * np.exp(self.brownian_process(n, mu, sigma))

    def correlated_process(self, process_func, corr, cov_mat=None,**kwargs):
        if corr and isinstance(corr, float): # if corr is a real number        
            z1 = process_func(**kwargs)
            z2 = process_func(**kwargs)
            return z1, corr * z1 + np.sqrt(1 - corr**2) * z2
        else: # if corr is not a real number
            # check if a covariance matrix is passed
            if cov_mat is None:
                raise ValueError("Either a corr value or a cov_mat must be passed.")
            try:
                L = np.linalg.cholesky(cov_mat)
            except:
                raise ValueError("Not a valid covariance matrix.")
            n = cov_mat.shape[0]
            
            z = np.array(
                [process_func(**kwargs) for _ in range(n)]
            )
            
            return np.dot(L, z)

    def correlated_brownian_process(
        self, 
        n: int, 
        corr=None,
        mu=0.1, 
        sigma=0.01, 
        cov_mat=None, 
        S0: float = 1
    ):
        return self.correlated_process(
            process_func=self.brownian_process, 
            corr=corr, 
            n=n,
            mu=mu,
            sigma=sigma, 
            cov_mat=cov_mat,
            S0=S0
        )

    def correlated_geometric_brownian_process(
        self, 
        n: int, 
        corr=None,  
        mu: float = 0.1, 
        sigma: float = 0.01,  
        cov_mat=None,
        S0: float = 1
    ):
        if cov_mat is None:
            z1, z2 = self.correlated_brownian_process(
                corr=corr,
                n=n,
                mu=mu, 
                sigma=sigma,
                cov_mat=cov_mat,
                S0=1
            )
            return S0*np.exp(z1), S0*np.exp(z2)
        else:
            return S0*np.exp(
                self.correlated_brownian_process(
                                corr=corr,
                                n=n,
                                mu=mu, 
                                sigma=sigma,
                                cov_mat=cov_mat,
                                S0=1
                            )
            )

    def add_seasonality(
        self,
        process,
        start_idx=None,
        amp=None,
        freq=None,
        diffusion=None,
        how="full_random",
        how_diffusion=None,
        contamination:float=0,
        random_seed=None,
        seasonality_limit:int=None
    ):
        if how not in ["full_random", "random_mag", "manual"]:
            warnings.warn("Invalid specification for arg 'how', default to full_random.")
            how = "full_random"

        if random_seed:
            np.random.seed(random_seed)
        if "random" in how:
            if how == "full_random":
                start_idx = np.random.choice(
                    list(range(len(process))),
                    size = 1
                )
                # if full_random, generate amp, freq, diffusion, contamination randomly
                # using process mean and std as benchmark
                std = np.std(process)
                amp = np.random.random() * std * 2 * np.random.choice([-1,1]) if not amp else amp
                freq = 1/len(process) * np.random.randint(10, 100) if not freq else freq
                diffusion = np.random.random() * std * 2 * np.random.choice([-1,1]) if not diffusion else diffusion
                how_diffusion = np.random.choice([None, "linear", "sqrt"]) if not how_diffusion else how_diffusion

            elif how == "random_mag":
                if not start_idx:
                    raise RuntimeError("Specified random_mag overlay but no start_idx is provided.")
                
        else:
            raise NotImplementedError(f"how = {how} not implemented.")

        helper = SimulationHelpers()
        seasonality = helper.gen_seasonality(
            n = len(process),
            amp=amp,
            freq=freq,
            contamination=contamination,
            how_diffusion=how_diffusion,
            diffusion=diffusion
        )

        if seasonality_limit is not None and isinstance(seasonality_limit, int):
            seasonality[seasonality_limit:] = 0

        return self.__overlay(process, seasonality)

    def get_random_z_above_thresh(self, thresh: float):
        sign = thresh > 0 - (thresh <= 0)
        abs_thresh = np.abs(thresh)
        p = norm.cdf(abs_thresh)
        norm_mult = norm.cdf(np.abs(np.random.normal())) - 0.5
        z = norm.ppf(p + norm_mult * (1 - p))
        return z * sign

    def get_random_t_above_thresh(self, thresh: float, df: int):
        distr = t(df)
        sign = thresh > 0 - (thresh <= 0)
        print("sign:", sign)
        abs_thresh = np.abs(thresh)
        print("abs_thresh:", abs_thresh)
        p = distr.cdf(abs_thresh)
        print("p:", p )
        mult = distr.cdf(np.abs(np.random.standard_t(df)))-0.5
        print("norm_mult:", mult)
        z = distr.ppf(p + mult * (1 - p))
        return z * sign

    def add_outlier(
        self,
        process,
        thresh=norm.ppf(0.95),
        # thresh_z = 10,
        how="full_random",
        ma_window=10,
        random_seed=None,
        count=1,
        super_process=None, 
        outlier_indices=None
    ):
        if how not in ["full_random", "random_mag", "manual"]:
            warnings.warn("Invalid specification for arg 'how', default to full_random.")
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
                if not outlier_indices:
                    raise RuntimeError("Specified random_mag overlay but no outlier_indices is provided.")
                for idx in outlier_indices:
                    if idx not in range(ma_window - 1, len(process)):
                        raise ValueError(f"Specified index {idx} out of valid range.")
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
            print(f"outlier added at indices {', '.join([str(idx) for idx in outlier_indices])}")
            outlier_z[outlier_indices] = [
                self.get_random_t_above_thresh(thresh, ma_window) for i in range(count)
                # thresh_z for i in range(count)
            ]
            
        
            super_process = outlier_z * pd.Series(process).rolling(ma_window).std()

            return self.__overlay(process, super_process)
        
        if how == "manual":
            if not super_process:
                raise RuntimeError("Specified manual overlay but no super_process is provided.")

            return self.__overlay(process, super_process)

    def add_regime_change(
        self, 
        process, 
        event_index, 
        shift, 
        regime_limit: int=None,
        perturb=False, 
        perturb_func=lambda x: 0
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

        if regime_limit:
            if event_index + regime_limit < len(super_process):
                super_process[event_index+regime_limit+1:] = 0
        
        return self.__overlay(process, super_process)

    def correlated_processes_with_correlated_outliers(
        self, 
        process_func, 
        corr, 
        cov_mat=None, 
        how="full_random",
        ma_window=10,
        random_seed=None,
        count=1,
        outlier_indices=None,
        **kwargs
    ):
        # Introduce similar outliers to independent series 
        # Then correlate them 
        if how == "full_random":
            outlier_indices = np.random.choice(
                list(range(ma_window - 1, len(process_func(**kwargs)))), size=count
            )
        elif how == "random_mag":
            if not outlier_indices:
                raise RuntimeError("Specified semi-random overlay but no outlier_indices is provided.")
            for idx in outlier_indices:
                if idx not in range(ma_window - 1, len(process_func(**kwargs))):
                    raise ValueError(f"Specified index {idx} out of valid range.")
            warnings.warn(
                (
                    "Specified random_mag with fixed indices"
                    f"{outlier_indices}. Argument 'count' overridden."
                )
            )
            count = len(outlier_indices)

        if corr and isinstance(corr, float): # if corr is a real number        
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
        else: # if corr is not a real number
            # check if a covariance matrix is passed
            if cov_mat is None:
                raise ValueError("Either a corr value or a cov_mat must be passed.")
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
                ) for process in z
            ]
            helper = SimulationHelpers()
            helper.plot(*z)
            z = np.array(
                z
            )
            
            return np.dot(L, z)

    def correlated_brownian_processes_with_CO(
        self, 
        n: int, 
        corr=None,
        cov_mat=None, 
        how="full_random",
        ma_window=10,
        random_seed=None,
        outlier_indices=None,
        count=1,
        S0:float = 1,
        mu:float=0.1, 
        sigma=0.01
    ):
        # write this by calling the function above.

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
            sigma=sigma
        )

    def correlated_geometric_brownian_processes_with_CO(
        self, 
        n: int, 
        corr=None,
        cov_mat=None, 
        how="full_random",
        ma_window=10,
        random_seed=None,
        outlier_indices=None,
        count=1,
        S0:float = 1,
        mu:float=0.1, 
        sigma=0.01
    ):
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
                S0=1
            )
            return S0*np.exp(z1), S0*np.exp(z2)
        else:
            return S0*np.exp(
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
                                S0=1
                            )
            )

if __name__ == "__main__":
    sim = BaseSimulation()

    def random_perturb(x: Iterable):
        return 0.0001 * (np.random.rand(len(x)) * 2 - 1) + 0.001 * (
            np.cos(x) + np.sin(x - 0.01)
        )


    def random_perturb_1(x: Iterable):
        # on geom brownian
        return 0.0001 * (np.random.rand(len(x)) * 2 - 1) + 0.001 * (
            np.cos(x) + np.sin(x - 0.01)
        )

    helper = SimulationHelpers()

    sigma = 0.01 
    Sig = helper.gen_rand_cov_mat(
        8, 
    )
    Sig /= 5
    print(Sig)
    data = sim.correlated_geometric_brownian_processes_with_CO(
        n=10000, mu = 0, sigma = 0.1, cov_mat=Sig, S0=100, 
        ma_window = 30, 
        how = "random_mag", 
        outlier_indices = [6000]
    )

    helper.plot(*data, func = "ret")

    # seasonality example use case:
    # p = sim.geom_brownian_process(2000, mu=0.1, sigma=0.1)
    # q = sim.add_seasonality(p, how = "full_random")
    # helper.plot(p, q)
