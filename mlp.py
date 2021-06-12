import numpy as np

"""
Running a Multilayer perceptron with only 1 activation function. Should the model have more than 1 act func, the code needs to be adjusted.
"""

def stochastic_gradient_descent(query, w, b, lr, act_funcs, target):
    print(">>>>> FORWARD PROPAGATION")
    history = forward_propagation(query, w, b, act_funcs, target)

    print("\n>>>>> BACKWARD PROPAGATION")
    backward_propagation(query, w, b, lr, act_funcs, history["z"], target, history["x"])

def forward_propagation(query, w, b, act_funcs, target):
    output = np.array(query[:])
    z = []
    x = [output]

    print("X0 = " + str(output))
    print()

    for i in range(len(w)):
        act_func = act_funcs[i]
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

def backward_propagation(query, w, b, lr, act_funcs, z, target, x):
    # Start by computing the delta of the last layer
    act_func = act_funcs[-1]
    #### WARNING: THIS ASSUMES A SSE ERROR FUNCTION
    # If it is not a SSE, switch the next line for the derivative of E in order to X
    error = act_func(z[-1]) - target
    if act_func != softmax:
        last_delta = np.array(error) * derivatives[act_func](z[-1])
        print("Delta" + str(len(z)) + " = dE/dX o dX/dZ (= derivative of the error function elementwise product derivative of the activation function)")
        print(str(np.array(error)) + " o " + str(derivatives[act_func](z[-1]))  + " = " + str(last_delta))
    else:
        last_delta = np.array(error)
        print("Delta" + str(len(z)) + " = dE/dZ (= derivative of the error function)")
        print("= " + str(last_delta))
        
    deltas = [last_delta]


    # Compute subsequent deltas
    for i in range(len(z) - 1, 0, -1):
        print()
        act_func = act_funcs[i]
        new_delta = np.array(w[i]).T @ np.array(deltas[-1]).reshape((len(deltas[-1]), 1)) * np.array(derivatives[act_func](z[i-1])).reshape((len(z[i-1]), 1))
        print("Delta" + str(i) + " = " + " dZ" + str(i+1) + "/dX" + str(i) + " . delta" + str(i+1) + " o dX" + str(i) + "/dZ" + str(i))
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
    return 0.1 * (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

def softmax(z):
    return np.exp(z) / sum(np.exp(z))

def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

def tanh_derivative(z):
    return 0.1 * (1 - np.power(np.tanh(z), 2))

def softmax_derivative(z):
    return z * (1 - z)


derivatives = dict()
derivatives[sigmoid] = sigmoid_derivative
derivatives[tanh] = tanh_derivative
derivatives[softmax] = softmax_derivative




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

W = [w1, w2]
b = [b1, b2]

query = [1, 1, 1, 1, 1]
target = [0, 0]

# WARNING: TANH HAS BEEN MULTIPLIED BY 0.1 (exam2)
stochastic_gradient_descent(query, W, b, 1, [tanh, tanh], target)

#query = [1, 0, 0, 0, 1]
#stochastic_gradient_descent(query, W, b, 1, sigmoid, target)
