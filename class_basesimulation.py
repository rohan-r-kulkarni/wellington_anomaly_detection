import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections.abc import Iterable
import warnings
from scipy.stats import norm
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
        start,
        amp,
        how="random",
        random_seed=None,
    ):
        if how not in ["random", "manual"]:
            warnings.warn("Invalid specification for arg 'how', default to random.")
            how = "random"

        super_process = np.zeros_like(process)
        # TODO: Write me after writing basic cross-correlation metric & multiple process generation   

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
        super_process=None, 
        list_indices=None
    ):
        if how not in ["random", "semi_random", "manual"]:
            warnings.warn("Invalid specification for arg 'how', default to random.")
            how = "random"

        super_process = np.zeros_like(process)
        outlier_z = np.zeros_like(process)
        if random_seed:
            np.random.seed(random_seed)
        if "random" in how:
            if how == "random":
                outlier_indices = np.random.choice(
                    list(range(ma_window - 1, len(process))), size=count
                )
            elif how == "semi_random":
                if not list_indices:
                    raise RuntimeError("Specified semi-random overlay but no list_indices is provided.")
                for idx in list_indices:
                    if idx not in range(ma_window - 1, len(process)):
                        raise ValueError(f"Specified index {idx} out of valid range.")
                outlier_indices = list_indices
            else:
                raise NotImplementedError(f"how = {how} not implemented.")

            global actual
            actual = outlier_indices
            print(f"outlier added at indices {', '.join([str(idx) for idx in outlier_indices])}")
            outlier_z[outlier_indices] = [
                self.get_random_z_above_thresh(thresh_z) for i in range(count)
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
    sigma = 0.02
    Sig = helper.gen_rand_cov_mat(
        3, 
        # sigma = sigma
    )
    print(Sig)
    data = sim.correlated_brownian_process(n=10000, mu = 0, cov_mat=Sig, S0=100)
    helper.plot(
        *data, 
        # func = "ret"
    )
    # np.linalg.cholesky(Sig)
    print(helper.corr_from_cov(Sig))
    