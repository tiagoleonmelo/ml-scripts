import math
from collections import Counter

"""
Decision Trees
    1. Calculate the initial entropy
    2. Compute entropy of each attribute
        2.1. Check what values can be assumed by that attribute
        2.2. Compute (weighted average) entropy of outputs
    3. Compute Gain (=E_initial - E_feature)
    4. Pick highest gain

"""

#################### ID3
def entropy_calc(outputs):
    c = Counter(outputs)
    entropy = 0

    fractions = []

    for k in c:
        fractions += [c[k] / len(outputs)]
        print("Frequency of '" + k + "': " + str(c[k]) + "/" + str(len(outputs)), end="; ")

    print()

    for f in fractions:
        entropy += (f * math.log(f,2))

    return round(-entropy, 3)

def entropy_of_feature(list_outputs):
    entropy = 0
    total_size = sum([len(o) for o in list_outputs])

    for output in list_outputs:
        temp = entropy_calc(output)
        print(">>> Entropy of " + str(output) + ": " + str(temp) + "; adding to weighted average with weight " + str(len(output)) + "/" + str(total_size))
        entropy += (len(output) / total_size) * temp

    return round(entropy, 3)

#################### CART
def impurity_calc(outputs):
    c = Counter(outputs)
    impurity = 1

    fractions = []

    for k in c:
        fractions += [c[k] / len(outputs)]
        print("Frequency of '" + k + "': " + str(c[k]) + "/" + str(len(outputs)), end="; ")

    print()

    for f in fractions:
        impurity -= f ** 2

    return round(impurity, 3)

def impurity_of_feature(list_outputs):
    entropy = 0
    total_size = sum([len(o) for o in list_outputs])

    for output in list_outputs:
        temp = impurity_calc(output)
        print(">>> Entropy of " + str(output) + ": " + str(temp) + "; adding to weighted average with weight " + str(len(output)) + "/" + str(total_size))
        entropy += (len(output) / total_size) * temp

    return round(entropy, 3)

#################### Driver Code
outputs = ['n', 't', 't', 'm', 'f']
init_entr = impurity_calc(outputs)
print("Initial Entropy: " + str(init_entr))
print()

outputs_f1 = [['t', 't', 'f'], ['n', 'm']]
f1_entr = impurity_of_feature(outputs_f1)
print("Entropy on feature f1: " + str(f1_entr))
print("Gain = E_initial - E_feature = " + str(init_entr) + " - " + str(f1_entr) + " = " + str(init_entr - f1_entr))
print()

outputs_f2 = [['n', 't'], ['t', 'm', 'f']]
f2_entr = impurity_of_feature(outputs_f2)
print("Entropy on feature f2: " + str(f2_entr))
print("Gain = E_initial - E_feature = " + str(init_entr) + " - " + str(f2_entr) + " = " + str(init_entr - f2_entr))
print()

outputs_f3 = [['n', 't', 'f'], ['t', 'm']]
f3_entr = impurity_of_feature(outputs_f3)
print("Entropy on feature f2: " + str(f3_entr))
print("Gain = E_initial - E_feature = " + str(init_entr) + " - " + str(f3_entr) + " = " + str(init_entr - f3_entr))
print()

outputs_f4 = [['n', 'm'], ['t', 't'], ['a']]
f4_entr = impurity_of_feature(outputs_f4)
print("Entropy on feature f2: " + str(f4_entr))
print("Gain = E_initial - E_feature = " + str(init_entr) + " - " + str(f4_entr) + " = " + str(init_entr - f4_entr))