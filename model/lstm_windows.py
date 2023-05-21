"""
Wellington Management Anomaly Detection, Spring 2023

This file utilizes the LSTM Autoencoder architecture from model/lstm_autoencoder.py
to implement brute force and STL-assisted LSTM anomaly detection in rolling windows.
This gives an OOP suite of methods to use for preprocessing, training, testing, analyzing, and plotting the LSTM windows methodology.
This file is not executable.
"""

import matplotlib.pyplot as plt
from model.lstm_autoencoder import DataGeneration, LSTM_Model_Base, reconstruction
from model.model_exec import get_outliers, lstm_run, reconstruction, temporalize
import numpy as np
import pandas as pd
import seaborn as sns
import sys
import tensorflow as tf

class LSTMWindows():
    """
    This class defines all model-related methods with the windows approach to anomaly detection using LSTMs.
    Training, testing, and window preprocessing are done with these methods.
    """

    def __init__(self, model, batch_size, epochs, seq_size, window_size, rolling_step, n_feature):
        """
        Defines parameters of the model training run.
        """

        self.model = model
        self.BATCH_SIZE = batch_size
        self.EPOCHS = epochs
        self.SEQ_SIZE = seq_size
        self.WINDOW_SIZE = window_size
        self.ROLLING_STEP = rolling_step
        self.N_FEATURE = n_feature
        
    def lstm_windows(self, train_data, test_data, train_only=False):
        """
        The main training and testig function for the LSTM windows approach. Temporalized training and test data must be fed in.

        :param train_data np.ndarray: temporalized training rolling windows data
        :param test_data np.ndarray: temporalized test window
        :param train_only bool: default False, whether or not to only do training of the model
        :return tuple: mean average error (MAE) of the reconstruction, the reconstruction time series, and the test window time series
        """
        history = self.model.fit(train_data, train_data,
                                    epochs=self.EPOCHS, batch_size=self.BATCH_SIZE)

        if train_only:
            return history

        pred = self.model(test_data)
        pred_reconstructed = reconstruction(pred, self.N_FEATURE)
        test_reconstructed = reconstruction(test_data, self.N_FEATURE)
        
        mae = tf.keras.losses.MeanAbsoluteError()

        return mae(pred_reconstructed,test_reconstructed).numpy(), pred_reconstructed, test_reconstructed

    def window_traintest(self, data, start, end):
        """
        This function temporalizes and generates the training and test window data for feeding into the lstm_windows() method.

        :param data np.ndarray: data to temporalize
        :param start int: start index of the test window, training temporalized data ends before this 
        :param end int: end index of the test window, no training or testing done after this
        :return tuple: the training and test window temporalized data
        """
        window_start = start
        window_end = end
        temporalize_before = temporalize(data[0:window_start], self.SEQ_SIZE)
        data_window_seq = temporalize(data[window_start:window_end], self.SEQ_SIZE)
        data_train_seq = temporalize_before

        return data_train_seq, data_window_seq

class LSTMWindowPlot():
    """
    This class defines all plotting methods needed for the LSTM windows analysis.
    Methods exist to read from files that results from previous training/testing sessions, 
    and to plot results in real time during training/testing in rolling windows.
    """

    def __init__(self):
        """
        For completion purposes only.
        """
        pass

    def read_from_file(self, bs, ep, ss, ws):
        """
        This file reads data from a previous saved training/testing LSTM windows session for plotting.

        :param bs int: the batch size of the session to load
        :param ep int: the epoch count of the session to load
        :param ss int: the sequence size of the session to load
        :param ws int: the window size of the session to load
        :return tuple: the anomalous indices, losses, list of window indices, reconstructions, and original time series of the session
        """
        filename = "windows" + str(ws) + "_ep" + str(ep) + "bs" + str(bs) + ".txt"
        file = open("lstm_windows_res/" + filename, "r")

        info = file.readlines()
        anomalous_ind = eval(info[0])
        losses = eval(info[1])
        windows = eval(info[2])
        reconstructs = eval(info[3])
        origs = eval(info[4])
        file.close()
        return tuple([anomalous_ind, losses, windows, reconstructs, origs])

    def window_loss_plot(self, reconstruct, orig, index = None, all = False, start=None, stop=None,  plot=True, ax=None, legend = False):
        """
        This is the main function for plotting the actual time series, reconstruction, and loss in a particular window

        :param reconstruct np.ndarray: the reconstruction to plot
        :param orig np.ndarray: the original window time series to plot
        :param index pd.Series: optional, default None, gives index labels on the x-axis for the dates of the window
        :param all bool: default False, whether or not to plot the entire time series
        :param start int: optional, default None, the starting index of the particular window
        :param stop int: optional, default None, the ending index of the particular window
        :param plot bool: default True, whether to plot the window results or just return the reconstruction error
        :param ax matplotlib.axes.Axes: optional, default None, the particular matplotlib Axes object to plot on
        :param legend bool: default True, whether or not to include a legend
        :return float: returns the MSE reconstruction error of the particular window
        """
        if not all:
            pred_window = reconstruct[start:stop][:,0]
            act_window = orig[start:stop][:,0]
        else:
            pred_window = reconstruct[:,0]
            act_window = orig[:,0]
            if start is None and stop is None:
                start = 0
                stop = len(reconstruct)

        if plot:
            if ax is None:
                plt.plot(pred_window, color="blue", label="Prediction")
                plt.plot(act_window, color="red", label="Actual")
                plt.fill_between(np.arange(0, stop-start), act_window, pred_window, color='coral')
                if index is not None:
                    index = index.iloc[start:stop]
                    plt.tick_params(axis='x', labelrotation=90)
                    plt.xticks(np.arange(0,len(index)), index)
                title = "Reconstruction Loss, Window = " + str(start) + "-" + str(stop)
                plt.title(title)
                plt.xlabel("Time")
                plt.ylabel("Value")
                plt.tight_layout()
                if legend:
                    plt.legend()
            else:
                ax.plot(pred_window, color="blue", label="Prediction")
                ax.plot(act_window, color="red", label="Actual")
                ax.fill_between(np.arange(0, stop-start), act_window, pred_window, color='coral')
                if index is not None:
                    index = index.iloc[start:stop]
                    ax.tick_params(axis='x', labelrotation=90)
                    ax.set_xticks(np.arange(0,len(index)), index)
                title = "Reconstruction Loss, Window = " + str(start) + "-" + str(stop)
                ax.set_title(title)
                ax.set_xlabel("Time")
                ax.set_ylabel("Value")
                plt.tight_layout()
                if legend:
                    ax.legend()        
        # we can now quantify the reconstruction loss in just this window
        return tf.get_static_value(tf.keras.losses.mse(pred_window, act_window))
    
    def plot_anomalous(self, data, format_sizetitle, std_dev = 1.5, anomindex = None, save = True, show=True):
        """
        A focused helper function that uses window_loss_plot to easily plot anomalous windows in particular.
        Widely used following the output of the read_from_file() method.

        :param data tuple: the anomalous indices, losses, list of window indices, reconstructions, and original time series of the session as given by read_from_file()
        :param format_sizetitle str: the title of the directory to save results in, if applicable
        :param std_dev float: default 1.5, the cut-off for z-scoring anomalous data beyond this multiple of the standard deviation
        :param anomindex pd.Series: optional, default None, index for the labels of the x-axis to feed into the 'index' parameter of window_loss_plot()
        :param save bool: default True, whether to save the figures in a particular directory
        :param show bool: default True, whether to show the plotted anomalous figures
        :return np.ndarray: the indices of anomalous values, returned for reference
        """
        loss = np.array(data[1])
        all_windows = np.array(data[2])
        reconstructs = np.array(data[3])[0].reshape(-1, 1)
        origs = np.array(data[4])[0].reshape(-1, 1)
        threshold = np.mean(loss) + std_dev*np.std(loss)
        
        anomalous_ind = [i for i, x in enumerate(loss > threshold) if x]
        for j in anomalous_ind:
            region = all_windows[j]
            plt.figure()
            self.window_loss_plot(reconstructs, origs, index = anomindex, start = region[0], stop=region[1], all=False, plot=True, legend=True)
            if save: 
                plt.savefig("lstm_windows_res/anom_plots/" + format_sizetitle + "/anom_" + str(region[0]) + ".png")
            if show:
                plt.show()

        return anomalous_ind
    
    def reg_hist(self, plot_data, sd_mult=2):
        """
        Plotting for the STL-assisted LSTM windows method. 
        This function generates the residual distribution plots and highlights anomalous residuals outside a given standard deviation multiplier.

        :param plot_data np.ndarray: residual data to histogram
        :param sd_mult float: standard deviation multiplier to denote threshold for z-scoring
        :return np.ndarray: returns boolean array of residuals that are beyond the threshold 
        """
        data_mean, data_std = plot_data.mean(), plot_data.std()
        cut_off = data_std * sd_mult
        lower, upper = data_mean - cut_off, data_mean + cut_off

        sns.displot(plot_data, bins=30, kde=True) #kernel density estimation
        plt.axvspan(xmin=lower, xmax=plot_data.min(), alpha=0.2, color='red')
        plt.axvspan(xmin=upper, xmax=plot_data.max(), alpha=0.2, color='red')
        plt.title("Resdiual Distribution, Highlighting Two Standard Deviations", fontsize=10)

        return abs(plot_data-data_mean) > cut_off #beyond two std devs
