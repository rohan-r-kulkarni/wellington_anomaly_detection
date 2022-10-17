from class_basesimulation import BaseSimulation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import keras


class DataGeneration:
    def __init__(self, total_time = 10000, seq_size = 1000):

        self.total_time = total_time
        self.seq_size = seq_size

        sim = BaseSimulation()
        self.data = sim.geom_brownian_process(n=total_time, mu=0.1, sigma=0.1)
        pass

    def to_sequences(self):
        sequences = []
        for i in range(len(self.data)-self.seq_size):
            sequences.append(self.data[i:i+self.seq_size])

        return np.array(sequences)

# class DataGeneration:
#     def __init__(self, size=300, time_steps=100):
#         self.size = size
#         self.data_normal = []
#         self.data_outlier = []
#
#         sim = BaseSimulation()
#         for i in range(size):
#             normal = sim.geom_brownian_process(n=time_steps, mu=0.1, sigma=0.1)
#             normal_outlier = pd.Series(sim.add_outlier(normal, count=2, thresh_z=3))
#
#             self.data_normal.append(normal)
#             self.data_outlier.append(normal_outlier)
#
#         self.data_normal = np.array(self.data_normal)
#         self.data_outlier = np.array(self.data_outlier)


class MyModel(tf.keras.Model):
    def __init__(self, seq_size):
        super().__init__()
        self.seq_size = seq_size
        self.lstm1 = tf.keras.layers.LSTM(128, activation = tf.nn.relu, return_sequences=True)
        self.lstm2 = tf.keras.layers.LSTM(64, activation = tf.nn.relu, return_sequences = False)
        self.repeat_v = tf.keras.layers.RepeatVector(self.seq_size)
        self.lstm3 = tf.keras.layers.LSTM(64, activation=tf.nn.relu, return_sequences=True)
        self.lstm4 = tf.keras.layers.LSTM(128, activation=tf.nn.relu, return_sequences=True)
        self.time_distribute = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1))

    def call(self, inputs):
        x = self.lstm1(inputs)
        x = self.lstm2(x)
        x = self.repeat_v(x)
        x = self.lstm3(x)
        x = self.lstm4(x)
        x = self.time_distribute(x)
        return x


if __name__ == "__main__":
    d = DataGeneration()
    data = d.data
    sequences = d.to_sequences()
    x = sequences.reshape(sequences.shape +(1,))


    # data_normal = d.data_normal
    # data_normal = data_normal.reshape(len(data_normal), len(data_normal[0]), 1)
    #
    # data_outlier = d.data_outlier
    # print(data_outlier)
    # data_outlier = data_outlier.reshape(len(data_outlier), len(data_outlier[0]), 1)
    #
    #
    model = MyModel(len(x[1]))
    model.compile(optimizer='adam', loss='mse')
    model.fit(x, x, epochs=20, batch_size = 500)
    model.save('tmp_model')
    model = keras.models.load_model('tmp_model')
    # print(model.summary())
    # print("done")

    # prediction = model.predict(data_normal)
    # print(data_normal)
    # print(prediction[0])
    # #prediction = model.predict(data_outlier[0])
    #
    # plt.plot(range(100), data_normal[0])
    # plt.plot(range(100), prediction[0])
    # plt.show()

    #print(model.call(data_normal))


