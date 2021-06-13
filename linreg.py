import pprint
import numpy as np

X = [0.0, 1.5, 3.0, 4.5, 6.0]
targets = [1, 0, -1, 0, 1]

def transformation(x, j):
    return [round(x**i, 2) for i in range(j)]

def sse_closed_form(X, targets):
    # w = ( X^T @ X ) ^1 @ X^T @ T
    return np.linalg.inv(np.array(X).T @ np.array(X)) @ np.array(X).T @ targets

def l2_reg_closed_form(X, targets, lamb):
    # w = ( X^T @ X + Lambda . I ) ^1 @ X^T @ T
    tempX = np.array(X)
    tempT = np.array(targets)

    print("About to be Inversed\n", tempX.T @ tempX + lamb * np.eye(len(X[0])))
    print()

    print("Inverse\n", np.linalg.inv(tempX.T @ tempX + lamb * np.eye(len(X[0]))))
    print()

    return np.linalg.inv(tempX.T @ tempX + lamb * np.eye(len(X[0]))) @ tempX.T @ tempT

design_mat = [transformation(x, 4) for x in X]
print("Design Matrix after applying transformation:")
pprint.pprint(design_mat)
print()

design_mat = [
    np.array([1] + [2, 4]),
    np.array([1] + [4, 2]),
    np.array([1] + [5, 6]),
    np.array([1] + [7, 5])
]

targets = [1, 1.5, 2, 2.5]

weights = sse_closed_form(design_mat, targets)
print("Computed Weights using the closed form solution w = ( X^T @ X ) ^1 @ X^T @ T")
pprint.pprint(weights)
print()

query = np.array([1, 2, 3])

print("Predicting", query, "by multiplying it with", weights)
print("X @ W =", weights @ query)
print()

l2_weights = l2_reg_closed_form(design_mat, targets, lamb=4)
print("Computed Weights using the L2 closed form solution w = ( X^T @ X + Lambda . I ) ^1 @ X^T @ T")
pprint.pprint(l2_weights)
print()

