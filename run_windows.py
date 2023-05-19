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
from sim_util.class_simulationhelper import SimulationHelpers
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

featured_credit = pd.read_csv(r'./data/featured_credit.csv', index_col="trans_date")
credit = featured_credit.loc[:,[col for col in featured_credit.columns if "_" not in col]]
d1 = featured_credit.loc[:,[col for col in featured_credit.columns if col.endswith("_d1")]]

# Select n companies with no zero observations and highest variances. 
credit_nozero = credit.loc[:,credit.apply(lambda x : (x==0).sum() == 0)]
N_COMPANIES = 4

companies = np.random.choice(credit_nozero.apply(lambda x: (x - x.mean())/x.std()).columns, N_COMPANIES, replace=False).tolist()
np.random.seed(None)

def standard_scale(x: pd.Series):
    return (x - x.mean())/x.std()

def has_substr_in_list(s:str, l:list):
    return not all(x not in s for x in l)

features = featured_credit.loc[:,[col for col in featured_credit if ("_" in col) and (has_substr_in_list(col, companies))]]
features = features.apply(standard_scale)

#train-test split, LSTM temporalize data
data = features.values[:,COMPANY_IND]
data = np.expand_dims(data,1)

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

#build the LSTMWindows and LSTMWindowPlot objects to run methods
lstm_windows = LSTMWindows(model, BATCH_SIZE, EPOCHS, SEQ_SIZE, WINDOW_SIZE, ROLLING_STEP, N_FEATURE)
lstm_window_plot = LSTMWindowPlot()

losses = []
reconstructs = []
origs = []

#moving windows over 10-1000 day range
for i in range(10,1000-WINDOW_SIZE+1, ROLLING_STEP):
    print(str(i) + ", " + str(i+WINDOW_SIZE))
    train, test = lstm_windows.window_traintest(data, i, i+WINDOW_SIZE)
    loss, reconstruct, original = lstm_windows.lstm_windows(train, test)
    losses.append(loss)
    reconstructs.append(reconstruct)
    origs.append(original)

#plot the loss histogram
plt.hist(losses, bins="auto")
plt.title("Reconstruction Losses, Window Size = " + str(WINDOW_SIZE))
plt.ylabel("No. of Windows")
plt.xlabel("Reconstruction Loss")
plt.show()

#visualize high reconstruction loss windows
threshold = np.mean(losses) + 2*np.std(losses) # beyond two std devs

wstarts = np.arange(10, 1000-WINDOW_SIZE+1, ROLLING_STEP)
windows = np.array(list(zip(wstarts, wstarts+WINDOW_SIZE)))
wdata = tuple([None, losses, windows, np.array(reconstructs).reshape(1,-1).tolist(), np.array(origs).reshape(1,-1).tolist()])

anomalous_ind = lstm_window_plot.plot_anomalous(wdata, "window_" + str(WINDOW_SIZE), save=False, show=True)

filename = "windows" + str(WINDOW_SIZE) + "_ep" + str(EPOCHS) + "bs" + str(BATCH_SIZE) + ".txt"
with open("lstm_windows_res/" + filename, "w") as outfile:
    outfile.write(str(anomalous_ind))
    outfile.write("\n")
    outfile.write(str(losses))
    outfile.write("\n")
    outfile.write(str(windows.tolist()))
    outfile.write("\n")

    outfile.write(str(np.array(reconstructs).reshape(1,-1).tolist())) 
    outfile.write("\n")
    outfile.write(str(np.array(origs).reshape(1,-1).tolist()))