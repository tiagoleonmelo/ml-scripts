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

def proj(X, vals, vecs):
    # Compute the projection according to the most relevant dimension
    idx = np.where(vals == max(vals))
    ret = []

    for x in X:
        ret += [round((vecs[idx] @ x)[0], 3)]

    return ret

X = [
    np.array([1, 1]),
    np.array([2, 2]),
    np.array([3, 3]),
    np.array([5, 4])
]

mean, cov = mean_and_cov(X)
print("Mean:", mean)
print("Covariance:", cov, sep="\n")
print()

eigvals, eigvecs = np.linalg.eig(cov)
print("Eigenvalues:", eigvals)
print("Eigenvectors (K-L transformation):", eigvecs, sep="\n")

print("\nProjecting the points according to the most significant dimension")
projections = proj(X, eigvals, eigvecs)
for x in range(len(projections)):
    print("X" + str(x+1) + ": " + str(projections[x]))