import math
import numpy as np

"""
mean = sum(elements) / len(elements)
stdev = sqrt(sum([(elem - mean)**2 for elem in elements]) / len(elements))
"""

X = [
    [170, 160],
    [80, 220],
    [90, 200],
    [60, 160],
    [50, 150],
    [70, 190],
]

targets = [
    'A',
    'B',
    'B',
    'A',
    'A',
    'B'
]

def prior(targets, classname):
    return round(len([t for t in targets if t == classname]) / len(targets), 3)

def normal(x, mean, stdev):
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * math.exp((-1 / (2 * stdev**2)) * (x - mean)**2)

def normal_multivar(x, mean, cov):
    elem1 = (1 / ((2 * math.pi)**(len(x)/2)))
    elem2 = 1 / math.sqrt(np.linalg.det(cov))
    elem3 = np.exp((-1 / 2) * np.array(x - mean).T @ np.linalg.inv(cov) @ np.array(x - mean))
    return elem1 * elem2 * elem3

def univariate(X, classname, targets):
    """
    Returns the average and standard deviation for each dimensions given a class
    """
    avg = []
    stdev = []

    for j in range(len(X[0])):
        temp_avg = 0
        for i in range(len(X)):
            if targets[i] == classname:
                temp_avg += X[i][j]
                
        temp_avg /= len([a for a in targets if a == classname])
                
        temp_stdev = 0
        for i in range(len(X)):
            if targets[i] == classname:
                temp_stdev += (X[i][j] - temp_avg)**2

        temp_stdev /= len([a for a in targets if a == classname]) - 1
        temp_stdev = math.sqrt(temp_stdev)

        avg += [round(temp_avg, 3)]
        stdev += [round(temp_stdev, 3)]

    return {"Averages": avg, "Standard Deviations": stdev}

def multivariate(X, classname, targets):
    mean = univariate(X, classname, targets)["Averages"]
    contrib = np.zeros((len(X[0]), len(X[0])))
    counter = 0

    for i in range(len(X)):
        if targets[i] == classname:
            elem1 = ((np.array(X[i]) - np.array(mean))).reshape(len(X[i]), 1)
            elem2 = (((np.array(X[i]) - np.array(mean))).reshape(len(X[i]), 1)).T
            contrib = contrib + elem1 @ elem2
            counter += 1

    return mean, contrib * (1 / (counter-1))

stats_a = univariate(X, 'A', targets)
stats_b = univariate(X, 'B', targets)

print("Stats for A:", stats_a)
print("Stats for B:", stats_b)

query = np.array([100, 225])

final_a = prior(targets, 'A') * normal(query[0], stats_a['Averages'][0], stats_a['Standard Deviations'][0]) * normal(query[1], stats_a['Averages'][1], stats_a['Standard Deviations'][1])
final_b = prior(targets, 'B') * normal(query[0], stats_b['Averages'][0], stats_b['Standard Deviations'][0]) * normal(query[1], stats_b['Averages'][1], stats_b['Standard Deviations'][1])

print("Final odds of A: " + str(final_a), "\nFinal odds of B: " + str(final_b))

print()
mean_a = multivariate(X, 'A', targets)[0]
cov_a = multivariate(X, 'A', targets)[1]
print("Multivariate stats for A:", "\n>> Mean: " + str(mean_a), "\n>> Covariance:\n" + str(cov_a))

mean_b = multivariate(X, 'B', targets)[0]
cov_b = multivariate(X, 'B', targets)[1]
print("\nMultivariate stats for B:", "\n>> Mean: " + str(mean_b), "\n>> Covariance:\n" + str(cov_b))
print()

print("Odds of multivar X belonging to A:", normal_multivar(query, mean_a, cov_a))
print("Odds of multivar X belonging to B:", normal_multivar(query, mean_b, cov_b))





