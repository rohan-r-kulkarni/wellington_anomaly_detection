import tensorflow as tf
import keras
import numpy as np
from scipy.stats import norm
from data_generation import gen_data
from pyod.utils import pairwise_distances_no_broadcast
from dense_autoencoder import DENSE_Model
from lstm_autoencoder import reconstruction


class OutlierMetric:
    """Class to organize outlier classification metric functions"""
    def quantile_outlier(self, l, thresh=0.05):
        """Decides that the top (100*thresh)% are outliers

        :param l: data
        :type l: Iterable
        :param thresh: top quantile cutoff
        :type thresh: float
        :rtype: List[int]
        """
        cutoff = np.quantile([np.abs(x) for x in l], 1 - thresh)
        return [idx for idx in range(len(l)) if np.abs(l[idx]) >= cutoff]

    def bb_outlier(self, l, thresh=0.05):
        """Decides that the observation k standard deviation outliers of mean
        are outliers, like a bollinger band.

        :param l: data
        :type l: Iterable
        :param thresh: used to determine std multiplier i.e. the observations with 
            probability <= 0.05 are decided as outliers.
        :type thresh: float
        :rtype: List[int]
        """
        std_mult = norm.ppf(1-thresh/2)
        ub = np.mean(l) + np.std(l) * std_mult
        lb = np.mean(l) - np.std(l) * std_mult
        return [idx for idx in range(len(l)) if l[idx] >= ub or l[idx] <= lb]

    def iqr_outlier(self, l):
        """Decides that the observations 1.5 IQR away from 75 quantile and 25
        quantile are outliers.

        :param l: data
        :type l: Iterable
        :rtype: List[int]
        """
        q75, q25 = np.percentile(l, [75, 25])
        iqr = q75 - q25
        return [
            idx
            for idx in range(len(l))
            if l[idx] >= q75 + 1.5 * iqr or l[idx] <= q25 - 1.5 * iqr
        ]


def lstm_run(
    model,
    train_data,
    test_data,
    batch_size,
    metric="top",
    thresh=0.05,
    epoch=80,
    early_stopping=False,
):
    """Execute lstm model.

    :param train_data: the training data
    :type train_data: np.ndarray
    :param test_data: the testing data
    :type test_data: np.ndarray
    :param batch_size: batch size
    :type batch_size: int
    :param metric: the metric to determine outlier
    :type metric: str
    :param thresh: probabiliy threshold for quantile outlier
    :type thresh: float
    :param epoch: number of epochs
    :type epoch: int
    :param early_stopping: if to employ early stop callback
    :type early_stopping: bool
    :rtype: (np.ndarray, Iterable)
    """
    metrics = OutlierMetric()
    # model training and prediction
    n_feature = model.n_feature
    seq_size = model.seq_size
    model.compile(optimizer="adam", loss="mse")
    if early_stopping:
        callback = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=10)
        model.fit(
            train_data,
            train_data,
            epochs=epoch,
            batch_size=batch_size,
            callbacks=[callback],
        )
    else:
        model.fit(train_data, train_data, epochs=epoch, batch_size=batch_size)
    model.save("lstm_model")

    # model prediction/reconstruction
    model = keras.models.load_model("lstm_model")
    pred = model.predict(test_data)

    test_reconstructed = reconstruction(test_data, n_feature)
    pred_reconstructed = reconstruction(pred, n_feature)

    distances = pairwise_distances_no_broadcast(test_reconstructed, pred_reconstructed)

    indx = {}
    n = int(len(test_reconstructed) * thresh)
    indx["top"] = np.argpartition(distances, -n)[-n:]
    indx["quantile"] = metrics.quantile_outlier(distances, thresh=thresh)
    indx["bb"] = metrics.bb_outlier(distances, thresh=thresh)
    indx["iqr"] = metrics.iqr_outlier(distances)

    if metric == "all":
        ind = indx
    else:
        ind = indx[metric]

    return pred_reconstructed, ind


# def dense_run(
#     train_data, test_data, n_feature, batch_size, metric="top", thresh=0.05, epoch=80
# ):
#     # model training and prediction
#     model = DENSE_Model(n_feature)
#     model.compile(optimizer="adam", loss="mse")
#     model.fit(train_data, train_data, epochs=epoch, batch_size=batch_size)
#     model.save("dense_model")

#     # model prediction/reconstruction
#     model = keras.models.load_model("dense_model")
#     pred = model.predict(test_data)
#     print(pred.shape, test_data.shape)

#     distances = pairwise_distances_no_broadcast(test_data, pred)

#     if metric == "top":
#         # top 1000 outliers
#         n = int(len(test_data) * thresh)
#         ind = np.argpartition(distances, -n)[-n:]
#     elif metric == "quantile":
#         ind = quantile_outlier(distances, thresh=thresh)
#     elif metric == "bb":
#         ind = bb_outlier(distances)

#     return pred, ind


def temporalize(X, seq_size):
    """Prepare input data for LSTM layers by slicing the data into sequences.

    :param seq_size: size of the look-back window
    :type seq_size: int
    :rtype: list
    """
    # break data into seq_size
    output_X = []

    for i in range(len(X) - seq_size + 1):
        output_X.append(X[i : i + seq_size, :])

    return np.array(output_X)


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

    total_time = 30000
    seq_size = 50
    n_feature = 3

    data = gen_data(10).to_numpy()

    partition_size = int(len(data) * 3 / 4)

    data_train = data[0:partition_size]
    data_test = data[partition_size:]

    data_train_seq = temporalize(data_train, seq_size)
    data_test_seq = temporalize(data_test, seq_size)

    lstm_pred, lstm_outliers = lstm_run(data_train_seq, data_test_seq)
    # dense_pred, dense_outliers = dense_run(data_train, data_test)

    print(f"Data test shape {data_test.shape}")
    print(f"lstm_pred shape {lstm_pred.shape}")
    # print(f"dense_pred shape {dense_pred.shape}")

    # plot the curves
    # TODO
