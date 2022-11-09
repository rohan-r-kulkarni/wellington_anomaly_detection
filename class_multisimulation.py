import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections.abc import Iterable
import warnings
from scipy.stats import norm, t
from class_simulationhelper import SimulationHelpers
from class_basesimulation import BaseSimulation

class MultiSimulation(BaseSimulation):

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
    multisim = MultiSimulation()
    helper = SimulationHelpers()
    Sig = helper.gen_rand_cov_mat(
        8, 
    )
    Sig /= 5
    print(Sig)
    data = multisim.correlated_geometric_brownian_processes_with_CO(
        n=10000, mu = 0, sigma = 0.1, cov_mat=Sig, S0=100, 
        ma_window = 30, 
        how = "random_mag", 
        outlier_indices = [6000]
    )

    helper.plot(*data, func="ret")
    plt.show()
