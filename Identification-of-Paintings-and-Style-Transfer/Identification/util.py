import sys
import os
import numpy as np
import cv2
import re


def load_image():
    filenames = list()
    for filename in os.listdir(sys.path[0]):
        filenames.append(filename)

    X1, X1names = list(), list()
    pat = re.compile(r'^(\d+\.)')
    for filename in filenames:
        if pat.search(filename):
            X1.append(cv2.imread(filename, 0))
            X1names.append(filename)

    X, X_test = list(), list()
    Xnames, Xnames_test = list(), list()
    y = list()  # labels
    pat_test = re.compile(r'^((1|7|10|20|23|25|26)\.)')
    pat_pos = re.compile(r'^((2|3|4|5|6|8|9|21|22|24|27|28)\.)')
    for i in range(len(X1names)):
        if pat_test.search(X1names[i]):
            X_test.append(X1[i])
            Xnames_test.append(X1names[i])
        else:
            X.append(X1[i])
            Xnames.append(X1names[i])
            if pat_pos.search(X1names[i]):
                y.append(1)
            else:
                y.append(0)
    return X, y, X_test, Xnames, Xnames_test


def distance(T0, C):
    """Compute the Euclidean distance."""
    dist = np.sqrt(np.sum((T0 - C) ** 2, 1))
    return dist


def resize(X, n=8):
    """
    Return:
         X: for x in X, x.shape == 2^n * 2^n.
    """
    Xnew = list()
    for x in X:
        res = cv2.resize(x, (pow(2, n), pow(2, n)),
                         interpolation=cv2.INTER_CUBIC)
        Xnew.append(res)
    return Xnew


def sp(X, y):
    Xnew = list()
    for x in X:
        a, b = x.shape
        a = int(a / 2)
        b = int(b / 2)
        Xnew.append(x[:a, :b])
        Xnew.append(x[:a, b:])
        Xnew.append(x[a:, :b])
        Xnew.append(x[a:, b:])
    ynew = list()
    for i in y:
        for j in range(4):
            ynew.append(i)
    return Xnew, ynew
