import numpy as np

X = [
    np.array([1] + [0, 0]),
    np.array([1] + [0, 2]),
    np.array([1] + [1, 1]),
    np.array([1] + [1, -1])
]

targets = [
    -1,
    1,
    1,
    -1
]

w = np.array([
    0,
    0,
    0
])

learning_rate = 1
epoch = 1

def sign(z):
    if z >= 0:
        return 1
    return -1

# Compute the output
# Calculate diff between output and target
# If its off use update rule:
# wi = wi + η (t − o) xi

while True:
    print("\n=========== Epoch " + str(epoch) + " ===========")
    updated = False
    for i in range(len(X)):
        dotprod = w @ X[i]
        o = sign(dotprod)
        numbers = "sign(" + str(w) + "." + str(X[i]) + ")"
        print("o(" + str(i + 1) + ") = sign(W . X(" + str(i+ 1) + ")) = " + numbers + " = sign(" + str(dotprod) + ") = " + str(o))

        error = targets[i] - o

        if error != 0:
            print("Error, updating weights: w = " + str(w) + " + " + str(learning_rate) + " * " + str(error) + " * " + str(X[i]), end=" = ")
            w = w + learning_rate * error * X[i]
            print(w)
            updated = True

    epoch += 1

    if not updated:
        break

print("\n>> Algorithm converged! Final Weights: " + str(w))