from feature import *
import util
import numpy as np

X, y, X_test, Xname, Xname_test = util.load_image()
y = np.array(y)
Theta = [0, np.pi / 6, np.pi / 3, np.pi / 2, 2 * np.pi / 3, 5 * np.pi / 6]
Sigma = [2, 3, 4, 5]

X = util.resize(X, 11)
X_test = util.resize(X_test, 11)

A, B = feature_gabor_list(X, Theta, Sigma)
T = gen_energy_array(A, B)

A_test, B_test = feature_gabor_list(X_test, Theta, Sigma)
T_test = gen_energy_array(A_test, B_test)

s = np.std(T, 0)
T1 = T / s
T_test1 = T_test / s


def svm(T, y, T_test):
    from sklearn.svm import SVC
    clf = SVC()
    clf.fit(T, y)
    result = clf.predict(T_test)
    return result


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

        s = np.std(T_train, 0)
        T_train = T_train / s
        T_valid = T_valid / s
        ans = svm(T_train, y_train, T_valid)
        ans = ans.reshape(1, -1)[0]
        print(ans)

        if abs(ans[0] - y_valid[0]) < 0.5:
            judge.append(1)
        else:
            judge.append(0)
    return np.array(judge)
