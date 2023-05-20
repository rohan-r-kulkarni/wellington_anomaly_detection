import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pymc as pm
from scipy.ndimage.interpolation import shift
import arviz as az
import seaborn as sns

def load_and_transform_company_data(data_file,min_max_data=False):
    cdf = pd.read_csv(data_file)
    cdf.columns=['company','date','data']
    cdf.date = pd.to_datetime(cdf.date)
    
    #Load the data, transform t into min-max scaled from 0 to 1
    t = (cdf["date"] - pd.Timestamp("1900-01-01")).dt.total_seconds().to_numpy()
    t_min = np.min(t)
    t_max = np.max(t)
    t = (t - t_min) / (t_max - t_min)
    cdf['t'] = t

    if min_max_data:
        cdf.data = (cdf.data - cdf.data.min())/(cdf.data.max()-cdf.data.min())
        
    return cdf

"""
Choose a company and split into train and test
"""
def run_decompose(cdf, company_id, split_date):
    data = cdf[cdf.company==cdf.company.unique()[company_id]].set_index('date')
    df_train = data[data.index < split_date]
    df_test = data[data.index >= split_date]
    return df_train, df_test


class BayesianModel:


    def __init__(self, **params):
        self.model_type = params['model_type']
        self.train_data = params['train_data']
        self.test_data = params['test_data']
        self.prior = None
        self.posterior = None
        self.post_data = None
        self.model = None
        self.test_posterior = None

        self.f_modes = 12

        if 'fourier_modes' in params:
            self.f_modes = params['fourier_modes']

        self.prior_alpha_mean = None
        if 'prior_alpha_mean' in params:
            self.prior_alpha_mean = params['prior_alpha_mean']

        self.prior_beta_mean = 0
        if 'prior_beta_mean' in params:
            self.prior_beta_mean = params['prior_beta_mean']

        self.prior_lag = 0
        if 'prior_lag' in params:
            self.prior_lag = params['prior_lag']

    
    def setup_model(self):
        df_train = self.train_data
        
        x_train = df_train['t']
        y_train = df_train.data

        df_train['date'] = df_train.index
        date_train = df_train["date"]
        data_train = df_train["data"]
        
        df = df_train
        
        #fourier modes
        n_order = self.f_modes
        periods = df_train["date"].dt.dayofyear / 365.25
        fourier_features = pd.DataFrame(
            {
                f"{func}_order_{order}": getattr(np, func)(2 * np.pi * periods * order)
                for order in range(1, n_order + 1)
                for func in ("sin", "cos")
            }
        )
        
        coords = {
            "fourier_features": np.arange(2 * n_order),
        }
        
        with pm.Model(check_bounds=False, coords=coords) as linear_with_seasonality:
            
            # data containers
            X = pm.MutableData("X", x_train)
            y = pm.MutableData("y", y_train)
            y_lagged = pm.MutableData("y_lagged", shift(y_train,1,cval=y_train.mean())) 
            
            X_four = pm.MutableData("fourier", fourier_features)

            if self.prior_alpha_mean is None:
                α = pm.Normal("α", mu=np.mean(y_train), sigma=np.std(y_train))
            else:
                α = pm.Normal("α", mu=self.prior_alpha_mean, sigma=np.std(y_train))
                
            β = pm.Normal("β", mu=self.prior_beta_mean, sigma=1)
            σ = pm.HalfCauchy("σ", beta=0.1)
            
            lag_coef = pm.Normal("lag_coefs", mu=self.prior_lag, sigma=1)
            
            β_fourier_yr = pm.Normal("β_fourier_yr", mu=0, sigma=10, dims="fourier_features")
            seasonality = pm.Deterministic(
                "seasonality", pm.math.dot(β_fourier_yr, X_four.T)
            )
            
            init = pm.Normal.dist(0, size=1)
            
            trend = pm.Deterministic("trend", α + β * X)
            ar = lag_coef * y_lagged
            
            #additive seasonality
            μ = trend + seasonality + ar
            
            pm.Normal("likelihood", mu=μ, sigma=σ, observed=y)
        
            self.model = linear_with_seasonality
    

    def sample_prior(self):
        with self.model:
            self.prior = pm.sample_prior_predictive(samples=1000)
        
        fig, ax = plt.subplots()
        df_train = self.train_data

        
        sns.lineplot(
            x="t", y="data", data=df_train, color="C0", label="data", ax=ax
        )
        az.plot_hdi(
            x=df_train['t'],
            y=self.prior.prior_predictive["likelihood"],
            hdi_prob=0.95,
            color="gray",
            smooth=False,
            fill_kwargs={"label": "HDI 50%", "alpha": 0.3},
            ax=ax,
        )
        az.plot_hdi(
            x=df_train['t'],
            y=self.prior.prior_predictive["likelihood"],
            hdi_prob=0.5,
            color="gray",
            smooth=False,
            fill_kwargs={"label": "HDI 95%", "alpha": 0.5},
            ax=ax,
        )
        ax.legend(loc="upper left")
        ax.set(title="Prior HDI Data Model");

    def sample_posterior(self):
        with self.model:
            temp_idata = pm.sample(
                target_accept=0.9, draws=100, chains=4
            )
            self.posterior = pm.sample_posterior_predictive(trace=temp_idata)
            self.post_data = temp_idata

        return self.post_data

    def plot_trace(self):
        az.plot_trace(
            data=self.post_data,
            compact=True,
            var_names=["β_fourier_yr",'α','β','lag_coefs','σ'],
            kind="rank_bars",
            backend_kwargs={"figsize": (12, 9), "layout": "constrained"},
        )

    def plot_posterior_insample_fit(self):
        fig, ax = plt.subplots()
        df_train = self.train_data
        temp_posterior_predictive = self.posterior
        sns.lineplot(
            x="t",
            y="data",
            data=df_train,
            marker="o",
            color="black",
            alpha=0.8,
            markersize=4,
            markeredgecolor="black",
            label="data (train)",
        )
        az.plot_hdi(
            x=df_train['t'],
            y=temp_posterior_predictive.posterior_predictive["likelihood"],
            hdi_prob=0.95,
            color="C0",
            smooth=False,
            fill_kwargs={"label": "HDI 50%", "alpha": 0.2},
            ax=ax,
        )
        az.plot_hdi(
            x=df_train['t'],
            y=temp_posterior_predictive.posterior_predictive["likelihood"],
            hdi_prob=0.5,
            color="C0",
            smooth=False,
            fill_kwargs={"label": "HDI 95%", "alpha": 0.3},
            ax=ax,
        )
        sns.lineplot(
            x=df_train['t'],
            y=temp_posterior_predictive.posterior_predictive["likelihood"]
            .stack(samples=("chain", "draw"))
            .mean(axis=1),
            marker="o",
            color="C0",
            markersize=4,
            markeredgecolor="C0",
            label="mean posterior predictive",
            ax=ax,
        )
        ax.legend(loc="upper left")
        ax.set(title="Posterior HDI Data Model (In-sample)");        

    def sample_test_data(self):
        # Update data reference.
        df_test = self.test_data
        x_test = df_test['t']
        y_test = df_test.data
        
        #fourier
        n_order = self.f_modes
        df_test['date'] = df_test.index
        periods = df_test["date"].dt.dayofyear / 365.25
        fourier_features_test = pd.DataFrame(
            {
                f"{func}_order_{order}": getattr(np, func)(2 * np.pi * periods * order)
                for order in range(1, n_order + 1)
                for func in ("sin", "cos")
            }
        )
        
        with self.model:
            pm.set_data({"X": x_test, 'y':y_test, 'fourier' : fourier_features_test, 'y_lagged': shift(y_test,1)})
            self.test_posterior = pm.sample_posterior_predictive(self.post_data, model=self.model)


    def plot_test_result(self):
        fig, ax = plt.subplots()
        df_test = self.test_data
        ppc_test = self.test_posterior
        sns.lineplot(
            x="t",
            y="data",
            data=df_test,
            marker="o",
            color="red",
            alpha=0.5,
            markersize=4,
            markeredgecolor="red",
            label="data (test)",
        )
        az.plot_hdi(
            x=df_test['t'],
            y=ppc_test.posterior_predictive["likelihood"],
            hdi_prob=0.95,
            color="C0",
            smooth=False,
            fill_kwargs={"label": "HDI 50%", "alpha": 0.2},
            ax=ax,
        )
        az.plot_hdi(
            x=df_test['t'],
            y=ppc_test.posterior_predictive["likelihood"],
            hdi_prob=0.5,
            color="C0",
            smooth=False,
            fill_kwargs={"label": "HDI 95%", "alpha": 0.3},
            ax=ax,
        )
        sns.lineplot(
            x=df_test['t'],
            y=ppc_test.posterior_predictive["likelihood"]
            .stack(samples=("chain", "draw"))
            .mean(axis=1),
            marker="o",
            color="C0",
            markersize=4,
            markeredgecolor="C0",
            label="mean posterior predictive",
            ax=ax,
        )
        ax.legend(loc="upper left")
        ax.set(title="OOS Posterior HDI Data Model");

       
        res_pred = np.array(ppc_test.posterior_predictive["likelihood"]
            .stack(samples=("chain", "draw"))
            .mean(axis=1))
        res_true = df_test.data

        return (res_true,res_pred)


        
        