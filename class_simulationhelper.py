import numpy as np
import pandas as pd
from collections.abc import Iterable
import matplotlib.pyplot as plt
from scipy.stats import norm

class SimulationHelpers:

    def plot(self, *args, figsize=(20, 14), func=None, row_lim = 4):
        n = len(args)
        fig, ax = plt.subplots(n//row_lim + 1, min(row_lim, n), sharey=True)
        for i, arg in enumerate(args):
            if isinstance(arg, Iterable):
                arg = [pd.Series(a) for a in arg]
            else:
                arg = [pd.Series(arg)]

            if func == "log":
                for a in arg:
                    np.log(a).plot(grid=True, figsize=figsize, 
                        ax=ax[i//row_lim,i%row_lim] if n > row_lim else ax[i%row_lim]
                    )
            elif func == "ret":
                for a in arg:
                    a.pct_change().plot(grid=True, figsize=figsize, 
                        ax=ax[i//row_lim,i%row_lim] if n > row_lim else ax[i%row_lim]
                    )
            elif func == "diff":
                for a in arg:
                    a.diff().plot(grid=True, figsize=figsize, 
                        ax=ax[i//row_lim,i%row_lim] if n > row_lim else ax[i%row_lim]
                    )
            else:
                for a in arg:
                    a.plot(grid=True, figsize=figsize, 
                        ax=ax[i//row_lim,i%row_lim] if n > row_lim else ax[i%row_lim]
                    )

    def gen_rand_cov_mat(self, n: int, random_seed=None, sigma=None):
        if random_seed is not None:
            np.random.seed(random_seed)
            
        A = np.random.rand(n,n)
        Cov = A.dot(A.T)
        if sigma is not None:
            if isinstance(sigma, float):
                Cov[np.diag_indices_from(Cov)] = sigma**2
            elif len(sigma) == n:
                Cov[np.diag_indices_from(Cov)] = np.array(sigma)**2
        return Cov

    def corr_from_cov(self, x):
        v = np.sqrt(np.diag(x))
        outer_v = np.outer(v, v)
        corr = x / outer_v
        corr[x == 0] = 0
        return corr

    def brownian_process(self, n: int, mu=0.1, sigma=0.01, S0=1):
        dt = 1 / n
        t = np.linspace(0, 1, n)
        W = np.random.standard_normal(size=n)
        W = np.cumsum(W) * np.sqrt(dt)  ### standard brownian motion ###
        X = (mu - 0.5 * sigma**2) * t + sigma * W
        return X*S0
        
    def standard_scaler(self, process):
        mean = np.mean(process)
        std = np.std(process)
        return (process - mean) / std

    def gen_seasonality(
        self,
        n,
        amp,
        freq,
        contamination,
        how_diffusion=None,
        diffusion=0.1,
    ):

        x = np.arange(0, n, 1)
        w = self.brownian_process(n, mu=0, sigma = contamination)
        if how_diffusion == "linear":
            amp = x * amp * diffusion
        elif how_diffusion == "sqrt":
            amp = np.sqrt(x) * amp * diffusion
        elif how_diffusion == "no":
            pass
        
        return amp * np.cos(x*freq) + w
