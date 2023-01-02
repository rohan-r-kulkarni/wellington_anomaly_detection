"""
IEOR4742 - Wellington Management Anomaly Detection, Fall 2022

contains EDA class SingleCompanyEDA for the EDA pipeline of a single company.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from arch.unitroot import PhillipsPerron
from statsmodels.tsa.seasonal import seasonal_decompose


class SingleCompanyEDA:
    """
    Conduct some exploratory data analysis on the data provided
    """

    def __init__(self, company: str, data: pd.DataFrame):
        """
        Read the data (data is the company's credit card transaction data)
        Get the first order differenced time series
        Store the STL decomposition results

        -----------------------------------------------------
        Parameters:

        - company: string
                   the name of the company
        - data: pandas.DataFrame
                the data (company's credit card transaction data)
        """

        self.company = company
        self.data = data[data["Unnamed: 0"] == self.company]
        self.data.set_index('trans_date', inplace=True)
        self.data.dropna(inplace=True)

        # store the STL decomposition results
        self.stlResults = None

        # first order differenced time series
        self.diff = self.data['data'].diff().dropna()

    def plot_original_time_series(self):
        """
        plot the original time series
        """

        self.data.plot()
        plt.title("Company"+self.company)

    def plot_acf(self):
        """
        plot the autocorrealtion function of the time series
        """

        sm.graphics.tsa.plot_acf(
            self.data['data'], title="Autocorrelation Plot of Company"+self.company)

    def plot_pacf(self):
        """
        plot the partial correaltion function of the time series
        """

        sm.graphics.tsa.plot_pacf(
            self.data['data'], title="Partial Correlation Plot of Company"+self.company)

    def stlDecompositon(self, period: int):
        """
        Conduct STL decomposition to the time series

        ---------------------------------------------------
        Parameters:

        - period: int
                  the length of the period for the seasonal part
        """

        self.stlResults = seasonal_decompose(self.data['data'], period=period)

    def plot_seasonal_part(self):
        """
        plot the seasonal part of the time series of the STL decomposition results
        """

        self.stlResults.seasonal.plot()

    def plot_trend_part(self):
        """
        plot the trend part of the time series of the STL decomposition results
        """

        self.stlResults.trend.plot()

    def plot_residual_part(self):
        """
        plot the residual part of the time series of the STL decomposition results
        """

        resid = self.stlResults.resid.reset_index()
        plt.scatter(resid['trans_date'], resid['resid'])

    def plot_stl(self):
        """
        plot the overall STL decomposition results of the time series
        """

        self.stlResults.plot()

    def pp_test(self, data: pd.Series):
        """
        Conduct the Phillips-Perron test on given data

        ----------------------------------------------------------------------------
        Parameters:

        -data:  pandas.Series
                It could be original data :self.data['data']
                It could also be first order differenced data self.diff
        """

        pp = PhillipsPerron(data)
        print(pp.summary().as_text())

    def boxplot(self, data: pd.Series):
        """
        Plot the boxplot of the given data

        ----------------------------------------------------------------------------
        Parameters:

        -data:  pandas.Series
                It could be original data :self.data['data']
                It could also be first order differenced data self.diff
        """

        sns.boxplot(data)

    def hist(self, data: pd.Series):
        """
        Plot the histogram of the given data
        Mark the data points which are three standard deviations away from the mean as red

        ----------------------------------------------------------------------------
        Parameters:

        -data:  pandas.Series
                It could be original data :self.data['data']
                It could also be first order differenced data self.diff
        """

        data_mean, data_std = data.mean(), data.std()
        cut_off = data_std * 3
        lower, upper = data_mean - cut_off, data_mean + cut_off

        sns.histplot(data, bins=70)
        plt.axvspan(xmin=lower, xmax=data.min(), alpha=0.2, color='red')
        plt.axvspan(xmin=upper, xmax=data.max(), alpha=0.2, color='red')
