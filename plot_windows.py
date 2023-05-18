import matplotlib.pyplot as plt
from model.lstm_windows import LSTMWindowPlot
import numpy as np
import pandas as pd
import sys
import tensorflow as tf

if len(sys.argv) == 1:
    DEFAULT_PLOT = True
    SAVE_NO_SHOW = True
    SHOW_RANGE = True
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
    BIN = 100

#build the LSTMWindowPlot object to run methods
lstm_window_plot = LSTMWindowPlot()

if DEFAULT_PLOT:
    WSIZES = [5, 10, 15, 20, 25, 50]
    wdata = np.empty(len(WSIZES), dtype=object)

    wdata[0] = lstm_window_plot.read_from_file(300, 10, SEQ_SIZE, 5)
    wdata[1] = lstm_window_plot.read_from_file(100, 10, SEQ_SIZE, 10)
    wdata[2] = lstm_window_plot.read_from_file(100, 10, SEQ_SIZE, 15)
    wdata[3] = lstm_window_plot.read_from_file(100, 10, SEQ_SIZE, 20)
    wdata[4] = lstm_window_plot.read_from_file(100, 10, SEQ_SIZE, 25)
    wdata[5] = lstm_window_plot.read_from_file(100, 10, SEQ_SIZE, 50)

    bin = 100

    for i in range(len(WSIZES)):
        plt.hist(wdata[i][1], bins=bin, label="Window Size = " + str(WSIZES[i]))

    plt.title("Reconstruction Losses varying Window Sizes")
    plt.ylabel("No. of Windows")
    plt.xlabel("Reconstruction Loss")
    plt.legend()
    if SAVE_NO_SHOW:
        plt.savefig("lstm_windows_res/hist_plots/hist_default.png")
    else:
        plt.show()

    #? Optional - show full range for each window size
    if SHOW_RANGE:
        for i in range(len(WSIZES)):
            reconstructs = np.array(wdata[i][3])[0].reshape(-1, 1)
            origs = np.array(wdata[i][4])[0].reshape(-1, 1)

            lstm_window_plot.window_loss_plot(reconstructs, origs, all=True, plot=True, legend=True)
            plt.show()
        
        for i in range(len(WSIZES)):
            lstm_window_plot.plot_anomalous(wdata[i], "window_" + str(WSIZES[i]), save=True, show=False)

else:
    w = lstm_window_plot.read_from_file(BATCH_SIZE, EPOCHS, SEQ_SIZE, WINDOW_SIZE)

    plt.hist(w[1], bins=BIN, label="Window Size = " + str(WINDOW_SIZE))

    plt.title("Reconstruction Losses varying Window Sizes")
    plt.ylabel("No. of Windows")
    plt.xlabel("Reconstruction Loss")
    plt.legend()
    plt.show()

    reconstructs = np.array(w[3])[0].reshape(-1, 1)
    origs = np.array(w[4])[0].reshape(-1, 1)

    lstm_window_plot.window_loss_plot(reconstructs, origs, all=True, plot=True, legend=True)
    plt.show()