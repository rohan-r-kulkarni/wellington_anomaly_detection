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

awindows = window_avg_resid[anomalous_resid].reset_index()
awindows = awindows.rename(columns={"date":"end_date"})
awindows = awindows.assign(start_date = awindows.end_date - pd.Timedelta(days=10)).loc[:, ["start_date", "end_date"]]

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

sys.exit()

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