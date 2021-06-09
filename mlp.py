import numpy as np

"""
Running a Multilayer perceptron with only 1 activation function. Should the model have more than 1 act func, the code needs to be adjusted.
"""

def stochastic_gradient_descent(query, w, b, lr, act_func, target):
    print(">>>>> FORWARD PROPAGATION")
    history = forward_propagation(query, w, b, act_func, target)

    print("\n>>>>> BACKWARD PROPAGATION")
    backward_propagation(query, w, b, lr, act_func, history["z"], target, history["x"])

def forward_propagation(query, w, b, act_func, target):
    output = np.array(query[:])
    z = []
    x = [output]

    print("X0 = " + str(output))
    print()

    for i in range(len(w)):
        print("=========== Layer " + str(i) + " ===========")
        print("Z" + str(i+1) + " = W" + str(i+1) + " @ X" + str(i) + " + b" + str(i+1) + " = " + str(w[i]) + " @ " + str(output) + " + " + str(b[i]))

        output = w[i] @ output + b[i]
        print("Z" + str(i+1) + " = " + str(output))
        z += [output]        

        # Adapt here if the act_func is variable
        output = act_func(output)
        print("X" + str(i+1) + " = " + str(output))
        x += [output]

        print()

    return {"z": z, "x": x}

def backward_propagation(query, w, b, lr, act_func, z, target, x):
    # Start by computing the delta of the last layer
    #### WARNING: THIS ASSUMES A SSE ERROR FUNCTION
    # If it is not a SSE, switch the next line for the derivative of E in order to X
    error = act_func(z[-1]) - target
    last_delta = np.array(error) * derivatives[act_func](z[-1])
    deltas = [last_delta]

    print("Delta" + str(len(z)) + " = dE/dX o dX/dZ (= derivative of the error function elementwise product derivative of the activation function)")
    print(str(np.array(error)) + " o " + str(derivatives[act_func](z[-1]))  + " = " + str(last_delta))

    # Compute subsequent deltas
    for i in range(len(z) - 1, 0, -1):
        print()
        new_delta = np.array(w[i]).T @ np.array(deltas[-1]).reshape((len(deltas[-1]), 1)) * np.array(derivatives[act_func](z[i-1])).reshape((len(z[i-1]), 1))
        print("Delta" + str(i) + " = " + " dZ" + str(i) + "/dX" + str(i) + " . delta" + str(i+1) + " o dX" + str(i) + "/dZ" + str(i))
        print(str(w[i]) + " . " + str(deltas[-1]) + " o " + str(derivatives[act_func](z[i-1]))  + " = " + str(new_delta.T[0]))
        deltas += [new_delta.T[0]]

    print("\n\n>>>>> UPDATE WEIGHTS")
    # Update weights
    for i in range(len(w)):
        print("=========== W " + str(i+1) + " ===========")
        print("Delta" + str(i), deltas[len(w) - i - 1].reshape((len(deltas[len(w) - i - 1]),1)), "\nX" + str(i), np.array(x[i]).reshape((len(x[i]),1)).T, "\n")
        print("LearningRate * Delta @ X =\n", lr * deltas[len(w) - i - 1].reshape((len(deltas[len(w) - i - 1]),1)) @ np.array(x[i]).reshape((len(x[i]),1)).T)
        print("Old Weight\n", w[i])
        w[i] = w[i] - lr * deltas[len(w) - i - 1].reshape((len(deltas[len(w) - i - 1]),1)) @ np.array(x[i]).reshape((len(x[i]),1)).T

        print("New Weight\n", w[i])
        print()

    # Update bias
    for i in range(len(b)):
        print("=========== B " + str(i+1) + " ===========")
        print("Delta" + str(i), deltas[len(b) - i - 1])
        print("LearningRate * Delta =\n", lr * deltas[len(b) - i - 1].reshape((len(deltas[len(b) - i - 1]),1)))
        print("Old Bias\n", b[i])
        b[i] = b[i] - lr * deltas[len(b) - i - 1]

        print("New Bias\n", b[i])
        print()

    return

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def tanh(z):
    return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

def tanh_derivative(z):
    return 1 - np.power(tanh(z), 2)


derivatives = dict()
derivatives[sigmoid] = sigmoid_derivative
derivatives[tanh] = tanh_derivative



w1 = [
    np.array([1, 1, 1, 1, 1]),
    np.array([1, 1, 1, 1, 1]),
    np.array([1, 1, 1, 1, 1])
]

b1 = np.array([0, 0, 0])

w2 = [
    np.array([1, 1, 1]),
    np.array([1, 1, 1])
]

b2 = np.array([0, 0])

w3 = [
    np.array([1, 1]),
    np.array([1, 1])
]

b3 = np.array([1, 1])

W = [w1, w2, w3]
b = [b1, b2, b3]

query = [1, 1, 1, 1, 1]
target = [1, -1]

stochastic_gradient_descent(query, W, b, 0.1, tanh, target)