import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

if len(sys.argv) != 5:
    print("ERROR: run with plot_windows.py BATCH_SIZE, EPOCHS, SEQ_SIZE, WINDOW_SIZE")
_, BATCH_SIZE, EPOCHS, SEQ_SIZE, WINDOW_SIZE = sys.argv
BATCH_SIZE = int(BATCH_SIZE)
EPOCHS = int(EPOCHS)
SEQ_SIZE = int(SEQ_SIZE)
WINDOW_SIZE = int(WINDOW_SIZE)

filename = "windows" + str(WINDOW_SIZE) + "_ep" + str(EPOCHS) + "bs" + str(BATCH_SIZE) + ".txt"
file = open("lstm_windows_res/" + filename, "r")

info = file.readlines()
anomalous_ind = eval(info[0])
losses = eval(info[1])
windows = eval(info[2])
reconstructs = eval(info[3])
origs = eval(info[4])

plt.hist(losses, bins="auto")
plt.title("Reconstruction Losses, Window Size = " + str(WINDOW_SIZE))
plt.ylabel("No. of Windows")
plt.xlabel("Reconstruction Loss")
plt.show()

file.close()