import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

import sys
if len(sys.argv) == 1:
    default = True
    SEQ_SIZE = 5
else:
    if len(sys.argv) != 5:
        print("ERROR: run with plot_windows.py BATCH_SIZE, EPOCHS, SEQ_SIZE, WINDOW_SIZE")
        sys.exit()
    default = False
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

def window_loss_plot(reconstruct, orig, all = False, start=None, stop=None,  plot=True, ax=None, legend = False):

    if not all:
        pred_window = reconstruct[start:stop][:,0]
        act_window = orig[start:stop][:,0]
    else:
        pred_window = reconstruct[:,0]
        act_window = orig[:,0]
        start = 0
        stop = len(pred_window)
        

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

def plot_anomalous(data, format_sizetitle, save = True, show=True):
    loss = np.array(data[1])
    all_windows = np.array(data[2])
    reconstructs = np.array(data[3])[0].reshape(-1, 1)
    origs = np.array(data[4])[0].reshape(-1, 1)
    threshold = np.mean(loss) + np.std(loss) # beyond a std dev

    anomalous_ind = [i for i, x in enumerate(loss > threshold) if x]
    for j in anomalous_ind:
        region = all_windows[j]
        plt.figure()
        window_loss_plot(reconstructs, origs, start = region[0], stop=region[1], all=False, plot=True, legend=True)
        if save: # modeling assumption: training occurs in the first 100 time series points
            plt.savefig("lstm_windows_res/anom_plots/" + format_sizetitle + "/anom_" + str(j) + ".png")
        if show:
            plt.show()

if default:
    wsizes = [5, 10, 15, 20, 25, 50]
    wdata = np.empty(len(wsizes), dtype=object)

    wdata[0] = read_from_file(300, 10, SEQ_SIZE, 5)
    wdata[1] = read_from_file(100, 10, SEQ_SIZE, 10)
    wdata[2] = read_from_file(100, 10, SEQ_SIZE, 15)
    wdata[3] = read_from_file(100, 10, SEQ_SIZE, 20)
    wdata[4] = read_from_file(100, 10, SEQ_SIZE, 25)
    wdata[5] = read_from_file(100, 10, SEQ_SIZE, 50)

    bin = 100

    for i in range(len(wsizes)):
        plt.hist(wdata[i][1], bins=bin, label="Window Size = " + str(wsizes[i]))

    plt.title("Reconstruction Losses varying Window Sizes")
    plt.ylabel("No. of Windows")
    plt.xlabel("Reconstruction Loss")
    plt.legend()
    plt.savefig("lstm_windows_res/hist_plots/hist_default.png")
    # plt.show()

    #show all range for each window size
    #? Optional
    # for i in range(len(wsizes)):
    #     reconstructs = np.array(wdata[i][3])[0].reshape(-1, 1)
    #     origs = np.array(wdata[i][4])[0].reshape(-1, 1)

    #     window_loss_plot(reconstructs, origs, all=True, plot=True, legend=True)
    #     # plt.show()
    
    for i in range(len(wsizes)):
        plot_anomalous(wdata[i], "window_" + str(wsizes[i]), save=True, show=False)

else:
    w = read_from_file(BATCH_SIZE, EPOCHS, SEQ_SIZE, WINDOW_SIZE)

    bin = 100

    plt.hist(w[1], bins=bin, label="Window Size = " + str(WINDOW_SIZE))

    plt.title("Reconstruction Losses varying Window Sizes")
    plt.ylabel("No. of Windows")
    plt.xlabel("Reconstruction Loss")
    plt.legend()
    plt.show()

    reconstructs = np.array(w[3])[0].reshape(-1, 1)
    origs = np.array(w[4])[0].reshape(-1, 1)

    window_loss_plot(reconstructs, origs, all=True, plot=True, legend=True)
    plt.show()





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