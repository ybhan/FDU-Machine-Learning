import numpy as np
from numpy import sqrt
import cv2


def gen_tau():
    """Generate filters tau."""
    tau = list(range(18))
    tau[0] = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16
    tau[1] = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]) / 16
    tau[2] = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]) / 16
    tau[3] = np.array([[1, 1, 0], [1, 0, -1], [0, -1, -1]]) * sqrt(2) / 16
    tau[4] = np.array([[0, 1, 1], [-1, 0, 1], [-1, -1, 0]]) * sqrt(2) / 16
    tau[5] = np.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]]) * sqrt(7) / 16
    tau[6] = np.array([[-1, 2, -1], [-2, 4, -2], [-1, 2, -1]]) / 48
    tau[7] = np.array([[-1, -2, -1], [2, 4, 2], [-1, -2, -1]]) / 48
    tau[8] = np.array([[0, 0, -1], [0, 2, 0], [-1, 0, 0]]) / 12
    tau[9] = np.array([[-1, 0, 0], [0, 2, 0], [0, 0, -1]]) / 12
    tau[10] = np.array([[0, 1, 0], [-1, 0, -1], [0, 1, 0]]) * sqrt(2) / 12
    tau[11] = np.array([[-1, 0, 1], [2, 0, -2], [-1, 0, 1]]) * sqrt(2) / 16
    tau[12] = np.array([[-1, 2, -1], [0, 0, 0], [1, -2, 1]]) * sqrt(2) / 16
    tau[13] = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]]) / 48
    tau[14] = np.array([[0, 0, 0], [-1, 2, -1], [0, 0, 0]]) * sqrt(2) / 12
    tau[15] = np.array([[-1, 2, -1], [0, 0, 0], [-1, 2, -1]]) * sqrt(2) / 24
    tau[16] = np.array([[0, -1, 0], [0, 2, 0], [0, -1, 0]]) * sqrt(2) / 12
    tau[17] = np.array([[-1, 0, -1], [2, 0, 2], [-1, 0, -1]]) * sqrt(2) / 24
    return tau


def feature_gtf(x, tau=gen_tau()):
    """Feature Extraction, via Geometric Tight Frame.
    Args:
        x: image matrix
        tau: filters
    Return:
        a: Geometric Tight Frame feature matrix
    """
    a = list(range(len(tau)))
    for i in range(len(tau)):
        a[i] = cv2.filter2D(x, -1, tau[i])
    return np.array(a)


def gen_sta(a):
    """Generate three statistics.
    Arg:
        a: feature matrix
    Return:
        t = [mu, sd, tail], where dim(t) = 18*3 = 54
    """
    t = list(range(len(a) * 3))
    for i in range(len(a)):
        t[i] = np.mean(a[i])
        t[i + len(a)] = np.std(a[i])
        t[i + len(a) * 2] = np.size(a[i][abs(a[i] - t[i]) > t[i + len(a)]])
    return np.array(t)


def feature_gtf_list(X):
    """Combine features of all images."""
    tau = gen_tau()
    A = list()
    for x in X:
        A.append(feature_gtf(x, tau))
    return A


def gen_sta_array(A):
    """Combine statistics of all images (including normalization).
    Arg:
        A: feature list
    Return:
        T: feature statistics
    """
    T = list()
    for a in A:
        T.append(gen_sta(a))
    T = np.array(T)
    return T


def normalize(T):
    """
    Returns:
        T: feature statistics (after normalization)
        mean: means of T
        std: standard deviations of T
    """
    mean = np.mean(T, 0)
    std = np.std(T, 0)
    T = (T - mean) / std
    return T, mean, std


def PCA(T, T_test):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=0.95)
    T_p = pca.fit_transform(T)
    T_test_p = pca.transform(T_test)
    return T_p, T_test_p, pca.explained_variance_ratio_


def gabor_fn(theta, sigma):
    sigma = sigma / np.sqrt(2)
    Lambda = sigma / np.pi
    sigma_x = sigma
    sigma_y = float(sigma)

    # Bounding box
    # nstds = 3  # Number of standard deviation sigma
    # xmax = max(abs(nstds * sigma_x * np.cos(theta)), abs(nstds * sigma_y * np.sin(theta)))
    # xmax = np.ceil(max(1, xmax))
    # ymax = max(abs(nstds * sigma_x * np.sin(theta)), abs(nstds * sigma_y * np.cos(theta)))
    # ymax = np.ceil(max(1, ymax))
    xmax = ymax = 64
    xmin = -xmax
    ymin = -ymax
    (y, x) = np.meshgrid(np.arange(ymin, ymax + 1), np.arange(xmin, xmax + 1))

    # Rotation 
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)

    Re = (np.exp(-.5 * (x_theta ** 2 / sigma ** 2 + y_theta ** 2 / sigma ** 2))
          * np.cos(2 * np.pi / Lambda * x_theta))
    Im = (np.exp(-.5 * (x_theta ** 2 / sigma ** 2 + y_theta ** 2 / sigma ** 2))
          * np.sin(2 * np.pi / Lambda * x_theta))
    return Re, Im, xmax, ymax


def feature_gabor(x, Theta, Sigma):
    """Feature Extraction, via gabor filter.
    Args:
        x: image matrix
    Return:
        a: feature matrix
    """
    a = list()
    b = list()
    for theta in Theta:
        for sigma in Sigma:
            Re, Im, xmax, ymax = gabor_fn(theta, sigma)
            a.append(cv2.filter2D(x, -1, Re))
            b.append(cv2.filter2D(x, -1, Im))
    return np.array(a), np.array(b)


def gen_energy(a, b):
    """Generate energy.
    Arg:
        a: feature matrix
    Return:
        t: energy
    """
    t = list()
    for i in range(24):
        t.append(np.sum(a[i] ** 2 / 1000 + b[i] ** 2 / 1000))
    return np.array(t)


def feature_gabor_list(X, Theta, Sigma):
    """Combine features of all images."""
    A, B = list(), list()
    for x in X:
        a, b = feature_gabor(x, Theta, Sigma)
        A.append(a)
        B.append(b)
    return A, B


def gen_energy_array(A, B):
    T = list()
    for i in range(len(A)):
        T.append(gen_energy(A[i], B[i]))
    T = np.array(T)
    return T
