# Misc Notes
Random stuff

## Hack for calculating covariance
(x1 - u)(x1 - u)^T

Compute for each x and then calculate the average (or adjusted average dividing by N - 1)

## Entropy
-(freq * log(freq) + ..)

## Impurity
1 - freq^2 - freq^2 - ...

## Solving Regressions
### Closed Form Solution
1. Derive the error expression (gradient of error in order to W -> [dE/dw0 dE/dW1 ...])
2. Equal it to 0
3. If we can have an expression of the form "w = something" then we have a Closed Form Solution.

### Gradient Descent
w = w - learning_rate * gradient_w(E)

Start by computing the gradient of activation function
>d(sigmoid(x))/dx = sigmoid(x) * (1 - sigmoid(x))
>d(tanh(x))/dx = 1 - tanh(x)**2

Now just compute the gradient of the error function

### RBF
Compute the RBF transformation (bias + 1 dim per centroid)
Now do grad desc:
>net = w.phi
>output = act_func(net)
>w = w + lr * (target - output) * phi