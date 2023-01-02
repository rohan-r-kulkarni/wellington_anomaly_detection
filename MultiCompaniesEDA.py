"""
IEOR4742 - Wellington Management Anomaly Detection, Fall 2022

contains EDA class MultiCompaniesEDA for the EDA pipeline of multiple companies.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from arch.unitroot import PhillipsPerron
from statsmodels.tsa.seasonal import seasonal_decompose


class MultiCompaniesEDA:
    """
    Combine multiple companies together to do the data analysis
    """

    def __init__(self, company_list: np.array, data: pd.DataFrame):
        """
        Read the data (data is the company's credit card transaction data)
        Get the first order differenced time series

        --------------------------------------------------------------------------
        Parameters:
        - company_list: numpy.array
                        the list of the company that you want to combine and conduct data analysis on
        - data: pandas.DataFrame
                a dataframe which contains the data (companies' credit card transaction data)
        """

        self.company_list = company_list
        self.companyseries = pd.DataFrame(data[(data["Unnamed: 0"].
                                                isin(company_list))][data.columns].
                                          dropna().
                                          groupby(['trans_date', 'Unnamed: 0'])['Unnamed: 0', 'data'].mean().unstack())

        # first ordered differenced time series of multiple companies
        self.companies_d1 = self.companyseries.diff().dropna()

    def plot_original_time_series(self, data: pd.DataFrame):
        """
        Plot the time series of multiple companies
        You can plot the original time series of multiple companies.
        Or you can plot the first ordered differenced time series of different companies.

        -----------------------------------------------------------------
        Parameters:

        - data: pandas.DataFrame
                It could be original data :self.companyseries
                It could also be first order differenced data self.companies_d1
        """

        data.plot()

    def plot_autocorrelation_among_companies(self, data: pd.DataFrame):
        """
        Plot the autocorrelation among multiple companies
        You can plot the autocorrelation of the original time series of multiple companies.
        Or you can plot the autocorrelation of the first ordered differenced time series of different companies.

        -----------------------------------------------------------------
        Parameters:

        - data: pandas.DataFrame
                It could be original data :self.companyseries
                It could also be first order differenced data self.companies_d1
        """

        pd.plotting.autocorrelation_plot(data)

    def plot_correlation_matrix(self, data: pd.DataFrame):
        """
        Plot the correlation matrix among multiple companies
        You can plot the correlation matrix of the original time series of multiple companies.
        Or you can plot the correlation matrix of the first ordered differenced time series of different companies.

        -----------------------------------------------------------------
        Parameters:

        - data: pandas.DataFrame
                It could be original data :self.companyseries
                It could also be first order differenced data self.companies_d1
        """

        data.corr().style.background_gradient(cmap='coolwarm')
