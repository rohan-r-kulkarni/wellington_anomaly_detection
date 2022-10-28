import numpy as np
import pandas as pd
from collections.abc import Iterable
import warnings
from scipy.stats import norm

class SimulationHelpers:

    def plot(self, *args, figsize=(20, 14), func=None, row_lim = 4):
        n = len(args)
        fig, ax = plt.subplots(n//row_lim + 1, min(row_lim, n), sharey=True)
        for i, arg in enumerate(args):
            arg = pd.Series(arg)
            if func == "log":
                np.log(arg).plot(grid=True, figsize=figsize, 
                    ax=ax[i//row_lim,i%row_lim] if n > row_lim else ax[i%row_lim]
                )
            elif func == "ret":
                arg.pct_change().plot(grid=True, figsize=figsize, 
                    ax=ax[i//row_lim,i%row_lim] if n > row_lim else ax[i%row_lim]
                )
            else:
                arg.plot(grid=True, figsize=figsize, 
                    ax=ax[i//row_lim,i%row_lim] if n > row_lim else ax[i%row_lim]
                )

    def gen_rand_cov_mat(self, n: int, sigma=None):
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
