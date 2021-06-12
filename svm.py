import numpy as np

def sign(z):
    if z >= 0:
        return 1
    return -1

def svm_dual_classif(X, targets, alphas, b, query):
    total = 0.0

    for i in range(len(X)):
        total += targets[i] * alphas[i] * X[i] @ query.T

    total += b

    print("Computing signal of", total)

    return sign(total)

def convert_alpha_to_w(X, targets, alphas):
    total = np.zeros(X[0].shape)

    for i in range(len(X)):
        total += targets[i] * alphas[i] * X[i]

    return total

def get_supp_vecs(X, targets, w, b):
    suppvecs = []

    for i in range(len(X)):
        temp = w @ X[i] + b
        print("Evaluating X", i +1, temp, targets[i])
        if temp == targets[i]:
            suppvecs += [X[i]]

    return suppvecs

X = [
    np.array([0, 0, 2]),
    np.array([0, 1, 8]),
    np.array([1, 0, 6]),
    np.array([1, 1, 7]),
    np.array([1, 1, 3])
]

targets = [1, 1, -1, -1, 1]

alphas = [0, 1, 0.5, 1, 0.5]

query = np.array([1, 1, 8])
b = -3

print("Number of Support Vectors:", len([_ for _ in alphas if _ != 0]))
print()

res = svm_dual_classif(X, targets, alphas, b, query)
print("Classification for vector", query, ":", res)
print()

w = convert_alpha_to_w(X, targets, alphas)
print("Weight vector", w)
print()

supvecs = get_supp_vecs(X, targets, w, 2)
print(supvecs)