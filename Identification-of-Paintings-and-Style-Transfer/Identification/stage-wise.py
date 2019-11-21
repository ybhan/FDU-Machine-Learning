import util
import feature
import numpy as np


def gen_center(T, y):
    """Generate the center of feature statistics of positive training set.
    Args:
        T: feature statistics list
        y: labels
    Return:
        C: center, where dim(C) = 18*3 = 54.
    """
    T_pos = [T[i] for i in range(len(y)) if y[i] == 1]
    C = np.mean(T_pos, 0).reshape(1, -1)
    return C


def feature_select(T_train, y_train, k=3):
    """Forward Stage-wise Principle features selection.
    Args:
        T_train: feature matrix list
        y_train: labels of training set
        k: the number of required principle features
    Return:
        B: principle features list
    """
    from sklearn.metrics import roc_auc_score

    T_principle_index = list()  # The index of Principle features
    AUC = list()

    # Forward Stage-wise algorithm
    for j in range(k):
        auc = list()
        Dist = list()

        for i in range(len(T_train.T)):
            if i in T_principle_index:
                auc.append(0)
                Dist.append(0)
                continue
            T_temp_index = T_principle_index + [i]
            T0 = T_train.T[T_temp_index].T  # transient statistics
            C0 = gen_center(T0, y_train)
            dist = util.distance(T0, C0)  # distance
            Dist.append(dist)
            auc.append(roc_auc_score(y_train, -dist))

        m = np.argmax(auc)
        T_principle_index = T_principle_index + [m]
        AUC.append(auc[m])

    dist = Dist[m]
    T_principle_index = np.array(T_principle_index)
    T_principle = T_train.T[T_principle_index].T

    return T_principle, T_principle_index, np.array(dist), AUC


def threshold(dist, y_train):
    table = np.vstack((dist, y_train)).T
    table = np.sort(table, 0)
    table = table[table[:, 0].argsort()]
    eps = list()
    for i in range(1, len(dist) - 1):
        i1 = np.arange(i)
        i2 = np.arange(i, len(dist))
        eps.append(sum(table[i1, 1] == 1) + sum(table[i2, 1] == 0))
    j = np.argmax(eps)
    eps[j] = 0
    j2 = np.argmax(eps)
    threshold = (table[j2, 0] + table[j, 0]) / 2
    return threshold


def cross_validation(T, y):
    """Use Cross Validation (leave-one-out) to select features.
    Args:
        T: feature statistics list
        y: labels
    """
    from sklearn.model_selection import LeaveOneOut
    y = np.array(y)
    judge = list()
    for train_index, valid_index in LeaveOneOut().split(T):
        T_train = T[train_index]
        T_valid = T[valid_index]
        y_train = y[train_index]
        y_valid = y[valid_index]

        T_train, mean, std = feature.normalize(T_train)
        T_principle, T_principle_index, dist, AUC = feature_select(T_train,
                                                                   y_train, k=3)
        ts = threshold(dist, y_train)
        C = gen_center(T_principle, y_train)
        T_valid = (T_valid - mean) / std
        dist_valid = util.distance(T_valid.T[T_principle_index].T, C)
        if y_valid[0] == 1:
            if dist_valid[0] < ts:
                judge.append(1)
            else:
                judge.append(0)
        else:
            if dist_valid[0] < ts:
                judge.append(0)
            else:
                judge.append(1)
    accuracy = sum(judge) / len(judge)
    return accuracy


def main():
    X, y, X_test, Xname, Xname_test = util.load_image()
    A = feature.feature_gtf_list(X)
    T = feature.gen_sta_array(A)
    accuracy = cross_validation(T, y)
    return accuracy


def predict(T, y, T_test):
    y = np.array(y)
    T, mean, std = feature.normalize(T)
    T_principle, T_principle_index, dist, AUC = feature_select(T, y, k=4)
    ts = threshold(dist, y)
    C = gen_center(T_principle, y)

    T_test = (T_test - mean) / std

    dist_test = util.distance(T_test.T[T_principle_index].T, C)
    judge = list()
    for d in dist_test:
        if d < ts:
            judge.append(1)
        else:
            judge.append(0)
    return np.array(judge), T_principle_index, AUC, dist


def cross_validation2(T, y):
    """Use Cross Validation (leave-one-out) to select features.
    Args:
        T: feature statistics list
        y: labels
    """
    from sklearn.model_selection import LeaveOneOut
    y = np.array(y)
    judge = list()
    T_principle_index = np.array([0, 18, 43])

    for train_index, valid_index in LeaveOneOut().split(T):
        T_train = T[train_index]
        T_valid = T[valid_index]
        y_train = y[train_index]

        T_train, mean, std = feature.normalize(T_train)

        T_principle = T_train.T[T_principle_index].T
        C = gen_center(T_principle, y_train)
        dist = util.distance(T_principle, C)
        ts = threshold(dist, y_train)

        T_valid = (T_valid - mean) / std
        dist_valid = util.distance(T_valid.T[T_principle_index].T, C)

        if dist_valid[0] < ts:
            judge.append(1)
        else:
            judge.append(0)
    return np.array(judge)


# X, y, X_test, Xname, Xname_test = util.load_image()
# A = feature_gtf_list(X)
# T = gen_sta_array(A)
#
# A_test = feature_gtf_list(X_test)
# T_test = gen_sta_array(A_test)

# acc = cross_validation2(T, y)
#
# p = predict(T, y, T_test)
# print(p)
# print(Xname_test)
