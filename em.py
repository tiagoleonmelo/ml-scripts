import numpy as np
import math

def normal_multivar(x, mean, cov):
    elem1= 1 / ((2 * math.pi)**(2*len(x)) * math.sqrt(np.linalg.det(cov)))
    elem2 = np.exp((-1 / 2) * np.array(x - mean).T @ np.linalg.inv(cov) @ np.array(x - mean))
    return elem1 * elem2

def expectation(X, means, covs, priors):
    print(">>>>>>>>>>>>> EXPECTATION")
    counter = 0
    posteriors = [[] for p in priors]
    for x in X:
        counter += 1
        joints = []
        print("============ Computing for X " + str(counter) + " ============")
        for i in range(len(means)):
            mean = means[i]
            cov = covs[i]
            prior = priors[i]
            print("Prior for class " + str(i) + " =", prior)

            likelihood = normal_multivar(x, mean, cov)
            print("Likelihood for class " + str(i) + " =", likelihood)

            joint = prior * likelihood
            joints += [joint]
            print("Joint Probability for class " + str(i) + " =", joint)

            print()

        # Normalized Posteriors
        for i in range(len(joints)):
            normalized = joints[i] / sum(joints)
            posteriors[i] += [normalized]
            print("Normalized Posterior for class " + str(i) + " =", normalized)

    return posteriors

def maximization(X, posteriors):
    print("\n>>>>>>>>>>>>> MAXIMIZATION")
    means = []
    new_covs = []
    new_priors = []

    for c in range(len(posteriors)):
        new_centroid = np.zeros(X[0].shape)

        for i in range(len(X)):
            new_centroid += posteriors[c][i] * X[i].T
    
        new_centroid /= sum(posteriors[c])
        means += [new_centroid]

        print("New Centroid for class " + str(c), new_centroid)

        new_cov = np.zeros(covs[0].shape)

        for i in range(len(X)):
            new_cov = new_cov + posteriors[c][i] * (np.array(X[i] - new_centroid).reshape((len(X[i]), 1)) @ np.array(X[i] - new_centroid).reshape((len(X[i]), 1)).T)

        new_cov = new_cov / sum(posteriors[c])
        new_covs += [new_cov]

        print("New Covariance Matrix for class " + str(c), new_cov, sep="\n")

        new_prior = sum(posteriors[c]) / sum([sum(p) for p in posteriors])

        print("New Prior for class " + str(c), new_prior)

        new_priors += [new_prior]


    return means, new_covs, new_priors

X = [
    np.array([2, 0]),
    np.array([1, 0]),
    np.array([0, 1])
]

means = [
    np.array([1, 0]),
    np.array([0, 1])
]

covs = [
    np.array([
        np.array([1, 0]),
        np.array([0, 1])
    ]),
    np.array([
        np.array([1, 0]),
        np.array([0, 1])
    ])
]

priors = [0.5, 0.5]

n_epochs = 2

for epoch in range(n_epochs):
    print("\nEPOCH ", epoch)
    posteriors = expectation(X, means, covs, priors)
    means, covs, priors = maximization(X, posteriors)