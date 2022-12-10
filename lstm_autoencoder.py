import keras
import numpy as np
import pandas as pd
import random
import tensorflow as tf
from class_basesimulation import BaseSimulation
from class_simulationhelper import SimulationHelpers
from collections.abc import Iterable
from pyod.utils.stat_models import pairwise_distances_no_broadcast
from typing import List


class DataGeneration:
    """Class created to fit data simulated by BaseSimulation to preliminary models
    :param total_time: number of observations to generate
    :type total_time: int
    :param seq_size: the size of the look-back window
    :type seq_size: int
    :param n_feature: number of features (dimensions)
    :type n_feature: int
    """

    def __init__(self, total_time: int = 10000, seq_size: int = 10, n_feature: int = 3):
        self.total_time = total_time  # length of simulation duration
        self.seq_size = seq_size  # length of looking back
        self.n_feature = n_feature
        self.sim = BaseSimulation()

    def to_sequences(self):
        seq_normal = []
        seq_outlier = []
        for i in range(len(self.data_normal) - self.seq_size):
            seq_normal.append(self.data_normal[i : i + self.seq_size])
            seq_outlier.append(self.data_outlier[i : i + self.seq_size])
        return np.array(seq_normal), np.array(seq_outlier)

    def multi_data(self):
        helper = SimulationHelpers()
        sigma = 0.02
        Sig = helper.gen_rand_cov_mat(
            self.n_feature,
            # sigma = sigma
        )

        data = self.sim.correlated_brownian_process(
            n=self.total_time, mu=0, cov_mat=Sig, S0=100
        ).T

        X = temporalize(X=data, seq_size=self.seq_size)

        X = np.array(X)
        X = X.reshape(X.shape[0], self.seq_size, self.n_feature)

        return X


class LSTM_Model_Base(tf.keras.Model):
    """LSTM AutoEncoder base class, which allows the specification of LSTM-AE Architecture
    via a list containing number of neurons per layers. e.g. [4,2,2,4] would specify
    a 2-layer LSTM-AE model with 4 neurons in the first layer and 2 neurons in the second
    layer of the encoder.

    :param seq_size: size of look-back window
    :type seq_size: int
    :param n_feature: number of features
    :type n_feature: int
    :param layers: the LSTM-AE Architecture. Must be symmetric, otherwise a runtime error
        is raised.
    :type layers: List[int]
    :param activation: activation function for each LSTM layer. Default to be tf.nn.tanh.
        Note that during our experimentations, tf.nn.tanh is the only working option.
    :type activation: Callable
    """

    def __init__(
        self,
        seq_size: int,
        n_feature: int,
        layers: List[int],
        activation=tf.nn.tanh,
        mid_activation=tf.nn.tanh,
    ):
        super().__init__()
        self.seq_size = seq_size
        self.n_feature = n_feature
        self.Layers = []

        # Check if layers present valid AutoEncoder architecture.
        def is_ae_valid(l: List[int]):
            """Checks if passed architecture is valid

            :param l: LSTM-AE architecture
            :type l: List[int]
            :rtype: bool
            """
            if len(l) % 2 != 0:
                return False
            mid = len(l) // 2
            for i in range(mid):
                if l[i] != l[-(i + 1)]:
                    return False
            return True

        # raise error if architecture passed in layers is not valid
        if not is_ae_valid(layers):
            raise RuntimeError("Not valid LSTM-AE architecture.")

        # if valid architecture, add layers to self.Layers
        for i in range(len(layers)):
            # if not at the end of adding encoder layers, just add layer
            # with return_sequences=True
            if i != len(layers) // 2 - 1:
                self.Layers.append(
                    tf.keras.layers.LSTM(
                        layers[i], activation=activation, return_sequences=True
                    )
                )
            # if adding the last layer before repeat layer, set return_sequence
            # to be False
            else:
                self.Layers.append(
                    tf.keras.layers.LSTM(
                        layers[i], activation=mid_activation, return_sequences=False
                    )
                )
                self.Layers.append(tf.keras.layers.RepeatVector(self.seq_size))

        self.Layers.append(
            tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_feature))
        )

    def call(self, inputs):
        """Standard tensorflow class call method"""
        x = inputs
        for layer in self.Layers:
            x = layer(x)
        return x


def temporalize(X, seq_size: int):
    """Prepare input data for LSTM layers by slicing the data into sequences.

    :param seq_size: size of the look-back window
    :type seq_size: int
    :rtype: list
    """
    output_X = []

    for i in range(len(X) - seq_size + 1):
        output_X.append(X[i : i + seq_size, :])

    return output_X


def reconstruction(seq_data, n_features):
    """Reconstruct the time series from sliced sequences.

    :param seq_data: temporalized data
    :type seq_data: np.ndarray
    :param n_features: number of features
    :type n_features: int
    :rtype: np.ndarray
    """
    multi = []
    for i in range(n_features):
        uni_seq = seq_data[:, :, i]

        uni = np.array([])
        j = 0

        for j in range(len(uni_seq)):
            uni = np.append(uni, uni_seq[j, 0])

        uni = np.append(uni, uni_seq[-1, 1:])
        multi.append(uni)

    multi = np.array(multi)
    return multi.T


if __name__ == "__main__":
    # system setup
    print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))
    print("Num GPUs Available: ", tf.config.list_physical_devices("GPU"))
    random.seed(10)
    helper = SimulationHelpers()

    # parameters
    total_time = 20000
    seq_size = 50
    n_feature = 3

    # data
    d = DataGeneration(total_time=total_time, seq_size=seq_size)
    x_normal = d.multi_data()

    # model training and prediction
    # model = LSTM_Model(seq_size, n_feature)
    # model.compile(optimizer='adam', loss='mse')
    # model.fit(x_normal, x_normal, epochs=80, batch_size=512)
    # model.save('tmp_model')

    # model prediction/reconstruction
    model = keras.models.load_model("tmp_model")
    pred = model.predict(x_normal)

    x_reconstructed = reconstruction(x_normal, n_feature, seq_size)
    pred_reconstructed = reconstruction(pred, n_feature, seq_size)

    distances = pairwise_distances_no_broadcast(x_reconstructed, pred_reconstructed)
    ind = np.argpartition(distances, -1000)[-1000:]

    print(ind)
    print(distances[ind])
    # plotting
    helper.plot(args=x_reconstructed.T, preds=pred_reconstructed.T, markers=ind)
    print("done")
