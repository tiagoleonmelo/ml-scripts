import numpy as np

def phi_calc(x, c):
    total = 0
    for i in range(len(x)):
        total += (x[i] - c[i])**2
        
    total /= 2
    total *= -1
    
    return np.exp(total)

phi1 = np.array([
        1,
        phi_calc(np.array([0, 0]), np.array([0.1, 0.1])),
        phi_calc(np.array([0, 0]), np.array([0, 1])),
        phi_calc(np.array([0, 0]), np.array([1, 0])),
        phi_calc(np.array([0, 0]), np.array([1.1, 1.1]))
    ])

phi2 = np.array([
        1,
        phi_calc(np.array([0, 1]), np.array([0.1, 0.1])),
        phi_calc(np.array([0, 1]), np.array([0, 1])),
        phi_calc(np.array([0, 1]), np.array([1, 0])),
        phi_calc(np.array([0, 1]), np.array([1.1, 1.1]))
    ])

phi3 = np.array([
        1,
        phi_calc(np.array([1, 0]), np.array([0.1, 0.1])),
        phi_calc(np.array([1, 0]), np.array([0, 1])),
        phi_calc(np.array([1, 0]), np.array([1, 0])),
        phi_calc(np.array([1, 0]), np.array([1.1, 1.1]))
    ])

phi4 = np.array([
        1,
        phi_calc(np.array([1, 1]), np.array([0.1, 0.1])),
        phi_calc(np.array([1, 1]), np.array([0, 1])),
        phi_calc(np.array([1, 1]), np.array([1, 0])),
        phi_calc(np.array([1, 1]), np.array([1.1, 1.1]))
    ])

phi5 = np.array([
        1,
        phi_calc(np.array([1.2, 1.2]), np.array([0.1, 0.1])),
        phi_calc(np.array([1.2, 1.2]), np.array([0, 1])),
        phi_calc(np.array([1.2, 1.2]), np.array([1, 0])),
        phi_calc(np.array([1.2, 1.2]), np.array([1.1, 1.1]))
    ])

phi6 = np.array([
        1,
        phi_calc(np.array([0.2, 0.2]), np.array([0.1, 0.1])),
        phi_calc(np.array([0.2, 0.2]), np.array([0, 1])),
        phi_calc(np.array([0.2, 0.2]), np.array([1, 0])),
        phi_calc(np.array([0.2, 0.2]), np.array([1.1, 1.1]))
    ])

Phi = np.array([
    phi1,
    phi2,
    phi3,
    phi4,
    phi5,
    phi6,
])

targets = np.array([1, 0, 0, 1, 1, 1])

w = np.array([1, 1, 1, 1, 1])

net = lambda x, w : x.dot(w)
sig = lambda net : 1 / (1 + np.exp(-net))

epochs = 2
lr = 1

for epoch in range(epochs):
    print("============ Epoch", epoch+1, "============")
    
    for i in range(len(Phi)):
        net_aux = net(w, Phi[i])
        print("net" + str(i+1), "=", net_aux)
        
        out = sig(net_aux)
        print("o" + str(i+1), "=", out)
        
        print("target" + str(i+1) + " - output" + str(i+1) + " =", targets[i] - out)
        w = w + lr * (targets[i] - out) * Phi[i]
        print("w_new =", w)
        
        print()
