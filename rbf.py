import numpy as np

def phi_calc(x, c, sigma=5):
    total = 0
    for i in range(len(x)):
        total += (x[i] - c[i])**2
        
    total /= (2 * sigma**2)
    total *= -1
    
    return np.exp(total)

x1 = np.array([1, 9])
x2 = np.array([0, 8])
x3 = np.array([1, 0])
x4 = np.array([1, 1])
x5 = np.array([0, -10])

c1 = np.array([1/2, 17/2])
c2 = np.array([1, 1/2])
c3 = np.array([0, -10])


phi1 = np.array([
        1,
        phi_calc(x1, c1, 5),
        phi_calc(x1, c2, 5),
        phi_calc(x1, c3, 5)
    ])

phi2 = np.array([
        1,
        phi_calc(x2, c1, 5),
        phi_calc(x2, c2, 5),
        phi_calc(x2, c3, 5)
    ])

phi3 = np.array([
        1,
        phi_calc(x3, c1, 5),
        phi_calc(x3, c2, 5),
        phi_calc(x3, c3, 5)
    ])

phi4 = np.array([
        1,
        phi_calc(x4, c1, 5),
        phi_calc(x4, c2, 5),
        phi_calc(x4, c3, 5)
    ])

phi5 = np.array([
        1,
        phi_calc(x5, c1, 5),
        phi_calc(x5, c2, 5),
        phi_calc(x5, c3, 5)
    ])

Phi = np.array([
    phi1,
    phi2,
    phi3,
    phi4,
    phi5
])

print("DESIGN MATRIX\n", Phi)

targets = np.array([-1, -1, 1, 1, -1])

w = np.array([1, 1, 1, 1])

net = lambda x, w : x.dot(w)
sig = lambda net : 1 / (1 + np.exp(-net))

def sign(z):
    if z >= 0:
        return 1
    return -1

epochs = 2
lr = 1

for epoch in range(epochs):
    print("============ Epoch", epoch+1, "============")
    
    for i in range(len(Phi)):
        net_aux = net(w, Phi[i])
        print("net" + str(i+1), "=", net_aux)
        
        out = sign(net_aux)
        print("o" + str(i+1), "=", out)
        
        print("target" + str(i+1) + " - output" + str(i+1) + " =", targets[i] - out)
        w = w + lr * (targets[i] - out) * Phi[i]
        print("w_new =", w)
        
        print()
