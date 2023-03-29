import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import tensorflow as tf
from sim_util.class_simulationhelper import SimulationHelpers
from model.lstm_autoencoder import DataGeneration, LSTM_Model_Base, reconstruction
from model.model_exec import get_outliers, lstm_run, reconstruction, temporalize


BATCH_SIZE = 10
EPOCHS = 30
SEQ_SIZE = 5 #lookback window for LSTM
WINDOW_SIZE = 10

company_ind = 1 #consider particular company
n_feature = 1

featured_credit = pd.read_csv(r'./data/featured_credit.csv', index_col="trans_date")
credit = featured_credit.loc[:,[col for col in featured_credit.columns if "_" not in col]]
d1 = featured_credit.loc[:,[col for col in featured_credit.columns if col.endswith("_d1")]]

# Select n companies with no zero observations and highest variances. 
credit_nozero = credit.loc[:,credit.apply(lambda x : (x==0).sum() == 0)]
n_companies = 4
np.random.seed(25)

#n_companies = 8
companies = np.random.choice(credit_nozero.apply(lambda x: (x - x.mean())/x.std()).columns, n_companies, replace=False).tolist()
np.random.seed(None)

def standard_scale(x: pd.Series):
    return (x - x.mean())/x.std()

def has_substr_in_list(s:str, l:list):
    return not all(x not in s for x in l)

features = featured_credit.loc[:,[col for col in featured_credit if ("_" in col) and (has_substr_in_list(col, companies))]]
features = features.apply(standard_scale)

#train-test split, LSTM temporalize data

data = features.values[:,company_ind]
data = np.expand_dims(data,1)
test_size = 0.4
partition_size = int(len(data) * (1 - test_size))

data_train = data[0:partition_size]
data_test = data[partition_size:]

data_train_seq = temporalize(data_train, SEQ_SIZE)
data_test_seq = temporalize(data_test, SEQ_SIZE)

#TODO: implement fully
def window_loss_plot(reconstruct, orig, all = False, start=None, stop=None,  plot=True, ax=None, legend = False):

    if not all:
        pred_window = reconstruct[start:stop][:,0]
        act_window = orig[start:stop][:,0]
    else:
        pred_window = reconstruct[:,0]
        act_window = orig[:,0]

    if plot:
        if ax is None:
            plt.plot(pred_window, color="blue", label="Prediction")
            plt.plot(act_window, color="red", label="Actual")
            plt.fill_between(np.arange(0, stop-start), act_window, pred_window, color='coral')
            title = "Reconstruction Loss, Window = " + str(start) + "-" + str(stop)
            plt.title(title)
            if legend:
                plt.legend()
        else:
            ax.plot(pred_window, color="blue", label="Prediction")
            ax.plot(act_window, color="red", label="Actual")
            ax.fill_between(np.arange(0, stop-start), act_window, pred_window, color='coral')
            title = "Reconstruction Loss, Window = " + str(start) + "-" + str(stop)
            ax.set_title(title)
            if legend:
                ax.legend()        
    # we can now quantify the reconstruction loss in just this window
    return tf.get_static_value(tf.keras.losses.mse(pred_window, act_window))

def window_traintest(start, end, SEQ_SIZE=5):
    window_start = start
    window_end = end
    temporalize_before = temporalize(data[0:window_start], SEQ_SIZE)
    data_window_seq = temporalize(data[window_start:window_end], SEQ_SIZE)
    temporalize_after = temporalize(data[window_end:], SEQ_SIZE)
    data_train_seq = np.concatenate((temporalize_before, temporalize_after), axis=0)

    return data_train_seq, data_window_seq

mae = tf.keras.losses.MeanAbsoluteError()


def lstm_windows(train_data, test_data, seq_size=SEQ_SIZE, n_feature = n_feature):
    
    model = LSTM_Model_Base(
        SEQ_SIZE, 
        n_feature, 
        [   128,
            64,
            64,
            128
        ], 
        mid_activation=tf.nn.tanh
    )
    n_feature = model.n_feature
    seq_size = model.seq_size
    model.compile(optimizer="adam", loss="mae")
    history = model.fit(train_data, train_data,
                                epochs=EPOCHS, batch_size=BATCH_SIZE)

    pred = model(test_data)
    pred_reconstructed = reconstruction(pred, n_feature)
    test_reconstructed = reconstruction(test_data, n_feature)

    return mae(pred_reconstructed,test_reconstructed).numpy(), pred_reconstructed, test_reconstructed

# window_loss_plot(data,np.random.rand(len(data)).reshape(-1,1),all=True,plot=True)
# plt.show()

losses = []
reconstructs = []
origs = []

for i in range(10,1000-WINDOW_SIZE, WINDOW_SIZE):
    print(str(i) + ", " + str(i+WINDOW_SIZE))
    train, test = window_traintest(i,i+WINDOW_SIZE)
    loss, reconstruct, original = lstm_windows(train, test)
    losses.append(loss)
    reconstructs.append(reconstruct)
    origs.append(original)


#plot the loss histogram
plt.hist(losses, bins="auto")
plt.title("Reconstruction Losses, Window Size = " + str(WINDOW_SIZE))
plt.ylabel("No. of Windows")
plt.xlabel("Reconstruction Loss")
plt.show()

#one standard dev
#visualize high reconstruction loss windows
threshold = np.mean(losses) + np.std(losses) # beyond a std dev
# threshold = np.mean(losses) - np.mean(losses)  #dummy

wstarts = np.arange(0, len(losses)*WINDOW_SIZE, WINDOW_SIZE)
windows = np.array(list(zip(wstarts, wstarts+WINDOW_SIZE)))
anomalous_ind = [i for i, x in enumerate(losses > threshold) if x]

#plot reconstructions of high loss windows
#window_loss_plot(reconstruct, orig, all = False, start=None, stop=None,  plot=True, ax=None, legend = False)
for i in anomalous_ind:
    region = windows[i]
    plt.figure()
    window_loss_plot(reconstructs[i], origs[i], start = region[0], stop=region[1], all=True, plot=True, legend=True)
    plt.show()


#two standard devs
#visualize high reconstruction loss windows
threshold = np.mean(losses) + 2*np.std(losses) # beyond a std dev
# threshold = np.mean(losses) - np.mean(losses)  #dummy

wstarts = np.arange(0, len(losses)*WINDOW_SIZE, WINDOW_SIZE)
windows = np.array(list(zip(wstarts, wstarts+WINDOW_SIZE)))
anomalous_ind = [i for i, x in enumerate(losses > threshold) if x]

#plot reconstructions of high loss windows
#window_loss_plot(reconstruct, orig, all = False, start=None, stop=None,  plot=True, ax=None, legend = False)
for i in anomalous_ind:
    region = windows[i]
    plt.figure()
    window_loss_plot(reconstructs[i], origs[i], start = region[0], stop=region[1], all=True, plot=True, legend=True)
    plt.show()