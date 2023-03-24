"""
IEOR4742 - Wellington Management Anomaly Detection, Fall 2022

This is the main model execution file that contains
    - utility class `OutlierMetric` that enables the classification of
        outliers using different methods.
    - other helper functions that enable the execution of LSTM-AE models,
        see docstrings for documentation.
"""

import numpy as np
import keras
import tensorflow as tf
import warnings
from sim_util.data_generation import gen_data
from model.lstm_autoencoder import reconstruction
from pyod.utils import pairwise_distances_no_broadcast
from scipy.stats import norm
from sklearn.metrics import pairwise_distances
from typing import Iterable


class OutlierMetric:
    """Class to organize outlier classification metric functions"""

    def quantile_outlier(self, l: Iterable, thresh=0.05):
        """Decides that the top (100*thresh)% are outliers

        :param l: data
        :type l: Iterable
        :param thresh: top quantile cutoff
        :type thresh: float
        :rtype: List[int]
        """
        cutoff = np.quantile([np.abs(x) for x in l], 1 - thresh)
        return [idx for idx in range(len(l)) if np.abs(l[idx]) >= cutoff]

    def bb_outlier(self, l: Iterable, thresh=0.05):
        """Decides that the observation k standard deviation outliers of mean
        are outliers, like a bollinger band.

        :param l: data
        :type l: Iterable
        :param thresh: used to determine std multiplier i.e. the observations with
            probability <= 0.05 are decided as outliers.
        :type thresh: float
        :rtype: List[int]
        """
        std_mult = norm.ppf(1 - thresh / 2)
        ub = np.mean(l) + np.std(l) * std_mult
        lb = np.mean(l) - np.std(l) * std_mult
        return [idx for idx in range(len(l)) if l[idx] >= ub or l[idx] <= lb]

    def iqr_outlier(self, l: Iterable):
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


def get_outliers(
    original: np.ndarray,
    prediction: np.ndarray,
    metric: str = "bb",
    cross_feature_check: bool = False,
    **kwargs
) -> np.ndarray:
    """Determine idiosyncratic outliers for each reconstructed feature.

    :param original: the original data
    :type original: np.ndarray
    :param prediction: the reconstructed data
    :type prediction: np.ndarray
    :param metric: the outlier metric from OutlierMetric to use for features.
        only "bb", "quantile" and "iqr" are valid.
    :type metric: str
    :param cross_feature_check: whether to check feature-specific outlier against
        the cross-feature outliers or not
    :type cross_feature_check: bool
    :rtype: np.ndarray
    """
    num_samples, num_features = original.shape
    pairwise_distances = np.square(original - prediction)

    m = OutlierMetric()
    metric_dict = {
        "bb": m.bb_outlier,
        "quantile": m.quantile_outlier,
        "iqr": m.iqr_outlier,
    }
    if metric not in metric_dict:
        warnings.warn("Invalid metric passed. Default to bb.")
        metric = "bb"
    outlier_func = metric_dict[metric]

    # if cross_feature_check flagged True, we check potential outliers of each feature
    # against all other features.
    # Important note: this must be done on standard-scaled, stationary time-series

    if cross_feature_check:
        mu = pairwise_distances.mean()
        sig = pairwise_distances.std()
        std_mult = norm.ppf(
            1 - kwargs['thresh'] / 2) if 'thresh' in kwargs else norm.ppf(1 - 0.1 / 2)
        ub, lb = mu + sig * std_mult, mu - sig * std_mult
        check_idx = np.array([
            [idx for idx in range(len(l)) if l[idx] >= ub or l[idx] <= lb]
            for l in pairwise_distances.T
        ])

        indices = []
        for i in range(num_features):
            indices.append(
                np.array(
                    [idx for idx in outlier_func(
                        pairwise_distances[:, i], **kwargs) if idx in check_idx[i]]
                )
            )
    else:
        indices = []
        for i in range(num_features):
            indices.append(outlier_func(pairwise_distances[:, i], **kwargs))

    return np.array(indices)


def lstm_run(
    model,
    train_data: np.ndarray,
    test_data: np.ndarray,
    batch_size: int,
    metric: str = "bb",
    thresh: float = 0.05,
    epoch: int = 80,
    early_stopping: bool = False,
    cross_feature_check: bool = False,

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
    :rtype: (np.ndarray, Iterable, Iterable)
    """
    metrics = OutlierMetric()
    # model training and prediction
    n_feature = model.n_feature
    seq_size = model.seq_size
    model.compile(optimizer="adam", loss="mse")
    if early_stopping:
        callback = tf.keras.callbacks.EarlyStopping(
            monitor="loss", patience=10)
        history = model.fit(
            train_data,
            train_data,
            epochs=epoch,
            batch_size=batch_size,
            callbacks=[callback],
        )
    else:
        history = model.fit(train_data, train_data,
                            epochs=epoch, batch_size=batch_size)
    model.save("lstm_model")

    # model prediction/reconstruction
    model = keras.models.load_model("lstm_model")
    pred = model.predict(test_data)

    test_reconstructed = reconstruction(test_data, n_feature)
    pred_reconstructed = reconstruction(pred, n_feature)

    ind = get_outliers(
        original=test_reconstructed,
        prediction=pred_reconstructed,
        metric=metric,
        thresh=thresh,
        cross_feature_check=cross_feature_check
    )

    return pred_reconstructed, ind, history


def temporalize(X, seq_size):
    """Prepare input data for LSTM layers by slicing the data into sequences.

    :param seq_size: size of the look-back window
    :type seq_size: int
    :rtype: list
    """
    # break data into seq_size
    output_X = []

    for i in range(len(X) - seq_size + 1):
        output_X.append(X[i: i + seq_size, :])

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
    x = np.random.random(size=[10, 10])
    x[1, 1] += 5
    print(x)
    print()
    print(x.std())
    print(x.mean())
    m = OutlierMetric()
    res = [m.bb_outlier(y) for y in x]
    print(res)
