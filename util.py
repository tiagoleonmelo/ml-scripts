import numpy as np

def mean_and_cov(X):
    mean = [0.0 for _ in range(len(X[0]))]

    for i in range(len(X)):
        for ii in range(len(X[0])):
            mean[ii] += X[i][ii]

    mean = np.array(mean)
    mean *= 1 / len(X)

    contrib = np.zeros((len(X[0]), len(X[0])))

    for i in range(len(X)):
        elem1 = ((np.array(X[i]) - np.array(mean))).reshape(len(X[i]), 1)
        elem2 = (((np.array(X[i]) - np.array(mean))).reshape(len(X[i]), 1)).T
        contrib = contrib + elem1 @ elem2

    ### WARNING: Adjust the following line according to adjusted cov/standard cov (remove -1 if the latter)
    contrib *= 1 / (len(X) -1)
    return mean, contrib

def det_and_inv(cov):
    return np.linalg.det(cov), np.linalg.inv(cov)

X = [
    np.array([2, -2]),
    np.array([1, 3]),
    np.array([0, -1]),
    np.array([2, 1])
]

mean, cov = mean_and_cov(X)
det, inv = det_and_inv(cov)
print("Mean:", mean)
print("Covariance:", cov, sep="\n")
print()

print("Determinant:", det)
print("Inverse:", inv, sep="\n")