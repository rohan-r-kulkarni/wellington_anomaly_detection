import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import tensorflow as tf

import sys
if len(sys.argv) == 1:
    SEQ_SIZE = 5
else:
    if len(sys.argv) != 5:
        print("ERROR: run with plot_windows.py BATCH_SIZE, EPOCHS, SEQ_SIZE, WINDOW_SIZE")
        sys.exit()
    _, BATCH_SIZE, EPOCHS, SEQ_SIZE, WINDOW_SIZE = sys.argv
    BATCH_SIZE = int(BATCH_SIZE)
    EPOCHS = int(EPOCHS)
    SEQ_SIZE = int(SEQ_SIZE)
    WINDOW_SIZE = int(WINDOW_SIZE)

def read_from_file(bs, ep, ss, ws):
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

w5 = read_from_file(300, 10, SEQ_SIZE, 5)
w10 = read_from_file(100, 10, SEQ_SIZE, 10)
w15 = read_from_file(100, 10, SEQ_SIZE, 15)
w20 = read_from_file(100, 10, SEQ_SIZE, 20)
w25 = read_from_file(100, 10, SEQ_SIZE, 25)
w50 = read_from_file(100, 10, SEQ_SIZE, 50)

# binslist = [25,50,75,100,150,200,250]
binslist = [100]

for bin in binslist:
    plt.hist(w5[1], bins=bin, label="Window Size = 5")
    plt.hist(w10[1], bins=bin, label="Window Size = 10")
    plt.hist(w15[1], bins=bin, label="Window Size = 15")
    plt.hist(w20[1], bins=bin, label="Window Size = 20")
    plt.hist(w25[1], bins=bin, label="Window Size = 25")
    plt.hist(w25[1], bins=bin, label="Window Size = 50")

    plt.title("Reconstruction Losses varying Window Sizes")
    plt.ylabel("No. of Windows")
    plt.xlabel("Reconstruction Loss")
    plt.legend()
    plt.show()

    #show beyond the majority
    plt.hist(w5[1], bins=bin, label="Window Size = 5")
    plt.hist(w10[1], bins=bin, label="Window Size = 10")
    plt.hist(w15[1], bins=bin, label="Window Size = 15")
    plt.hist(w20[1], bins=bin, label="Window Size = 20")
    plt.hist(w25[1], bins=bin, label="Window Size = 25")
    plt.hist(w25[1], bins=bin, label="Window Size = 50")

    plt.title("Reconstruction Losses varying Window Sizes")
    plt.ylabel("No. of Windows")
    plt.xlabel("Reconstruction Loss")
    plt.legend()
    plt.xlim([0.02, 1.4])
    plt.ylim([0, 10])
    plt.show()

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

#! fix saving of reconstruct data, not saved properly!!
# for wind in [w5, w10, w15]:
#     for i in wind[0]: #anomalous_ind
#         region = wind[2][i] #windows
#         reconstructs = wind[3]
#         print(reconstructs)
#         origs = wind[4]
#         print(origs)
#         plt.figure()
#         window_loss_plot(reconstructs[i], origs[i], start = region[0], stop=region[1], all=True, plot=True, legend=True)
#         plt.show()