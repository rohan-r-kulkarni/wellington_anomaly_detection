"""
Wellington Management Anomaly Detection, Spring 2023

This file utilizes the LSTM Autoencoder architecture to implement brute force LSTM anomaly detection.
Dataset used: credit card transaction dataset, company 1 from data/featured_credit.csv
Run with python3 run_windows.py BATCH_SIZE, EPOCHS, SEQ_SIZE, WINDOW_SIZE
Results are saved to lstm_windows_res.
"""

import keras
import logging
import matplotlib.pyplot as plt
from model.lstm_autoencoder import DataGeneration, LSTM_Model_Base, reconstruction
from model.lstm_windows import LSTMWindows, LSTMWindowPlot
from model.model_exec import get_outliers, lstm_run, reconstruction, temporalize
import numpy as np
import os
import pandas as pd
import random
import seaborn as sns
from sim_util.class_simulationhelper import SimulationHelpers
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
import sys
import tensorflow as tf

np.random.seed(25)

if len(sys.argv) != 5:
    print("ERROR: run with python3 run_windows.py BATCH_SIZE, EPOCHS, SEQ_SIZE, WINDOW_SIZE")
    sys.exit()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Set logging level to suppress INFO messages
logging.getLogger('tensorflow').setLevel(logging.WARN) 

"""
standard settings: 

BATCH_SIZE = 10
EPOCHS = 30
SEQ_SIZE = 5 #lookback window for LSTM
WINDOW_SIZE = 10
"""

_, BATCH_SIZE, EPOCHS, SEQ_SIZE, WINDOW_SIZE = sys.argv
BATCH_SIZE = int(BATCH_SIZE)
EPOCHS = int(EPOCHS)
SEQ_SIZE = int(SEQ_SIZE)
WINDOW_SIZE = int(WINDOW_SIZE)
ROLLING_STEP = 5
N_FEATURE = 1
COMPANY_IND = 1 #consider particular company
SHOW_NOT_SAVE = False

print("Running...")

sizetitle = "window_" + str(WINDOW_SIZE)
map_m_to_q = {
    1 : 1, 2: 1, 3: 1, 4:2, 5: 2, 6: 2, 7: 3, 8:3 ,9:3, 10: 4, 11:4, 12:4
}
dir = "data/"
cdf = pd.read_csv(dir+"data4columbia_credit.csv")
cdf.columns=['company','date','data']
cdf.date = pd.to_datetime(cdf.date)
cdf['month'] = cdf.date.apply(lambda x : x.month)
cdf['quarter'] = cdf.month.apply(lambda x : map_m_to_q[x])

data = cdf[cdf.company==cdf.company.unique()[COMPANY_IND]].set_index('date')
cid = data.company.iloc[0]
stl = seasonal_decompose(data['data'], period=365)
dtindex = pd.Series(data.index).dt.date

if SHOW_NOT_SAVE:
    stl.plot()

residuals = pd.DataFrame(stl.resid[~stl.resid.isna()])
residuals.plot(title="Residuals")
plt.ylabel("Value")
fig = plt.gcf()
fig.tight_layout()
if SHOW_NOT_SAVE:
    plt.show()
else:
    plt.savefig("lstm_windows_res/plots/" + sizetitle + "/resid_plot.png")
    plt.close()

window_avg_resid = residuals.rolling(WINDOW_SIZE).mean().dropna() #drop NA means rolling ends on last day

def reg_hist(plot_data):
    data_mean, data_std = plot_data.mean(), plot_data.std()
    cut_off = data_std * 2
    lower, upper = data_mean - cut_off, data_mean + cut_off

    sns.displot(plot_data, bins=30, kde=True) #kernel density estimation
    plt.axvspan(xmin=lower, xmax=plot_data.min(), alpha=0.2, color='red')
    plt.axvspan(xmin=upper, xmax=plot_data.max(), alpha=0.2, color='red')
    plt.title("Resdiual Distribution, Highlighting Two Standard Deviations", fontsize=10)

    return abs(plot_data-data_mean) > cut_off #beyond two std devs

anomalous_resid = reg_hist(window_avg_resid.loc[:, "resid"].values)
fig = plt.gcf()
fig.tight_layout()
if SHOW_NOT_SAVE:
    plt.show()
else:
    plt.savefig("lstm_windows_res/plots/" + sizetitle + "/resid_displot.png")
    plt.close()


awindows = window_avg_resid[anomalous_resid].reset_index()
awindows = awindows.rename(columns={"date":"end_date"})
awindows = awindows.assign(start_date = awindows.end_date - pd.Timedelta(days=WINDOW_SIZE)).loc[:, ["start_date", "end_date"]]

def get_resid_from_dates(x):
    start = x[0]
    end = x[1]
    start_index = np.where(residuals.index == start)[0][0]
    end_index = np.where(residuals.index == end)[0][0]

    return residuals.iloc[start_index:end_index, :]
    
replot_dfs = awindows.apply(get_resid_from_dates, axis=1)
residuals.plot()
for anom in replot_dfs:
    plt.plot(anom.index, anom.values.reshape(1,-1)[0], color="red")
plt.title("Residual Time Series, Highlighting Two Standard Deviations", fontsize=10)
plt.ylabel("Value")
fig = plt.gcf()
fig.tight_layout()
if SHOW_NOT_SAVE:
    plt.show()
else:
    plt.savefig("lstm_windows_res/plots/" + sizetitle + "/anom_resid_plot.png")
    plt.close()


# plot transactions
transactions = data.loc[data.company==cid, ["data"]]
transactions.plot()
for anom in replot_dfs:
    this_anom = transactions.loc[anom.index]
    plt.plot(anom.index, this_anom.values.reshape(1,-1)[0], color="red")
plt.title("Credit Card Time Series, Highlighting High z-score Residual Regions")
plt.ylabel("Value")
fig = plt.gcf()
fig.tight_layout()
if SHOW_NOT_SAVE:
    plt.show()
else:
    plt.savefig("lstm_windows_res/plots/" + sizetitle + "/anom_timeseries_plot.png")
    plt.close()

#loss, reconstruct, orig for each anomalous window
anom_windows = []
losses = []
reconstructs = []
origs = []

model = LSTM_Model_Base(
            SEQ_SIZE, 
            N_FEATURE, 
            [   128,
                64,
                64,
                128
            ], 
            mid_activation=tf.nn.tanh
        )
model.compile(optimizer="adam", loss="mae")
lstm_windows = LSTMWindows(model, BATCH_SIZE, EPOCHS, SEQ_SIZE, WINDOW_SIZE, ROLLING_STEP, N_FEATURE)
lstm_window_plot = LSTMWindowPlot()

checkpoint = 10
for x in awindows.values:
    start = x[0]
    end = x[1]
    start_index = np.where(residuals.index == start)[0][0]
    end_index = np.where(residuals.index == end)[0][0]
    anom_windows.append(tuple([start_index, end_index]))
    print()
    print( "########### ", "NEW ITERATION: ", start_index, "-", end_index, "CHECKPOINT: ", checkpoint, " ###########")

    #train up till the window
    lstm_data = np.expand_dims(transactions,1)
    lstm_data = transactions.values.reshape(-1,1)

    #train up to the test window from last checkpoint
    for i in range(checkpoint,start_index-WINDOW_SIZE+1, ROLLING_STEP):
        train, test = lstm_windows.window_traintest(lstm_data, i, i+WINDOW_SIZE)
        lstm_windows.lstm_windows(train, test, train_only=True)

    #test on the given window
    train, test = lstm_windows.window_traintest(lstm_data, start_index, end_index)
    loss, reconstruct, original = lstm_windows.lstm_windows(train, test)
    losses.append(loss)
    reconstructs.append(reconstruct)
    origs.append(original)

    checkpoint = start_index #reset checkpoint to start_index for training next window
   
    lstm_window_plot.window_loss_plot(reconstruct, original, index=dtindex, start=start_index, stop=end_index, all=True, plot=True, legend=True)
    fig = plt.gcf()
    fig.tight_layout()

    if SHOW_NOT_SAVE:
        plt.show()
    else:
        plt.savefig("lstm_windows_res/anom_plots/" + sizetitle + "/stl_ast/anom_" + str(start_index) + ".png")
        plt.close()