import numpy as np
import keras
import tensorflow as tf
import warnings
from data_generation import gen_data
from lstm_autoencoder import reconstruction
from pyod.utils import pairwise_distances_no_broadcast
from scipy.stats import norm
from sklearn.metrics import pairwise_distances


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
        std_mult = norm.ppf(1 - thresh / 2)
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


def get_outliers(
    original: np.ndarray, prediction: np.ndarray, metric: str = "bb", **kwargs
) -> np.ndarray:
    """Determine idiosyncratic outliers for each reconstructed feature.

    :param original: the original data
    :type original: np.ndarray
    :param prediction: the reconstructed data
    :type prediction: np.ndarray
    :param metric: the outlier metric from OutlierMetric to use for features.
        only "bb", "quantile" and "iqr" are valid.
    :type metric
    :rtype: np.ndarray
    """
    num_samples, num_features = original.shape
    pairwise_distances = np.square(original - prediction)

    m = OutlierMetric()
    metric_dict = {
        "bb": m.bb_outlier, 
        "quantile": m.quantile_outlier,
        "iqr": m.iqr_outlier
    }
    if metric not in metric_dict:
        warnings.warn("Invalid metric passed. Default to bb.")
        metric = "bb"
    
    outlier_func = metric_dict[metric]

    indices = []
    for i in range(num_features):
        indices.append(outlier_func(pairwise_distances[:, i], **kwargs))

    return np.array(indices)


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
    :rtype: (np.ndarray, Iterable, Iterable)
    """
    metrics = OutlierMetric()
    # model training and prediction
    n_feature = model.n_feature
    seq_size = model.seq_size
    model.compile(optimizer="adam", loss="mse")
    if early_stopping:
        callback = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=10)
        history = model.fit(
            train_data,
            train_data,
            epochs=epoch,
            batch_size=batch_size,
            callbacks=[callback],
        )
    else:
        history = model.fit(train_data, train_data, epochs=epoch, batch_size=batch_size)
    model.save("lstm_model")

    # model prediction/reconstruction
    model = keras.models.load_model("lstm_model")
    pred = model.predict(test_data)

    test_reconstructed = reconstruction(test_data, n_feature)
    pred_reconstructed = reconstruction(pred, n_feature)

    ind = get_outliers(test_reconstructed, pred_reconstructed)

    # print(f"Distances: {pairwise_distances(test_reconstructed, pred_reconstructed)}")
    #
    # indx = {}
    # n = int(len(test_reconstructed) * thresh)
    # indx["top"] = np.argpartition(distances, -n)[-n:]
    # indx["quantile"] = metrics.quantile_outlier(distances, thresh=thresh)
    # indx["bb"] = metrics.bb_outlier(distances, thresh=thresh)
    # indx["iqr"] = metrics.iqr_outlier(distances)
    #
    # if metric == "all":
    #     ind = indx
    # else:
    #     ind = indx[metric]

    return pred_reconstructed, ind, history


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
    pass
    # # system setup
    # print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))
    # print("Num GPUs Available: ", tf.config.list_physical_devices("GPU"))

    # # prepare dataset
    # featured_credit = pd.read_csv(r"data\featured_credit.csv", index_col="trans_date")
    # credit = featured_credit.loc[
    #     :, [col for col in featured_credit.columns if "_" not in col]
    # ]
    # d1 = featured_credit.loc[
    #     :, [col for col in featured_credit.columns if col.endswith("_d1")]
    # ]

    # # Select n companies with no zero observations and highest variances.
    # credit_nozero = credit.loc[:, credit.apply(lambda x: (x == 0).sum() == 0)]
    # # np.random.seed(42)
    # # n_companies = 6
    # np.random.seed(25)

    # n_companies = 8
    # companies = np.random.choice(
    #     credit_nozero.apply(lambda x: (x - x.mean()) / x.std()).columns,
    #     n_companies,
    #     replace=False,
    # ).tolist()
    # np.random.seed(None)

    # def standard_scale(x: pd.Series):
    #     return (x - x.mean()) / x.std()

    # def has_substr_in_list(s: str, l: list):
    #     return not all(x not in s for x in l)

    # features = featured_credit.loc[
    #     :,
    #     [
    #         col
    #         for col in featured_credit
    #         if ("_" in col) and (has_substr_in_list(col, companies))
    #     ],
    # ]
    # features = features.apply(standard_scale)
    # features.shape

    # # total_time = 30000
    # # seq_size = 25
    # seq_size = 5
    # n_feature = features.shape[1]

    # data = features.values
    # test_size = 0.4
    # partition_size = int(len(data) * (1 - test_size))

    # data_train = data[0:partition_size]
    # data_test = data[partition_size:]

    # data_train_seq = temporalize(data_train, seq_size)
    # data_test_seq = temporalize(data_test, seq_size)

    # lstm_pred, lstm_outliers = lstm_run(
    #     LSTM_Model_Base(
    #         seq_size, n_feature, [128, 64, 64, 128], mid_activation=tf.nn.tanh
    #     ),
    #     data_train_seq,
    #     data_test_seq,
    #     batch_size=512,
    #     epoch=300,
    #     metric="all",
    #     early_stopping=False,
    # )
    # # dense_pred, dense_outliers = dense_run(data_train, data_test, n_feature, batch_size = 100)

    # print(f"Data test shape {data_test.shape}")
    # print(f"lstm_pred shape {lstm_pred.shape}")
    # # print(f"dense_pred shape {dense_pred.shape}")
