import matplotlib.pyplot as plt
from model.lstm_autoencoder import DataGeneration, LSTM_Model_Base, reconstruction
from model.model_exec import get_outliers, lstm_run, reconstruction, temporalize
import numpy as np
import pandas as pd
import sys
import tensorflow as tf

class LSTMWindows():
    def __init__(self, model, batch_size, epochs, seq_size, window_size, rolling_step, n_feature):
        self.model = model
        self.BATCH_SIZE = batch_size
        self.EPOCHS = epochs
        self.SEQ_SIZE = seq_size
        self.WINDOW_SIZE = window_size
        self.ROLLING_STEP = rolling_step
        self.N_FEATURE = n_feature
        
    def lstm_windows(self, train_data, test_data, train_only=False):
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
        window_start = start
        window_end = end
        temporalize_before = temporalize(data[0:window_start], self.SEQ_SIZE)
        data_window_seq = temporalize(data[window_start:window_end], self.SEQ_SIZE)
        temporalize_after = temporalize(data[window_end:], self.SEQ_SIZE)
        data_train_seq = temporalize_before

        return data_train_seq, data_window_seq

class LSTMWindowPlot():

    def __init__(self):
        pass

    def read_from_file(self, bs, ep, ss, ws):
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
