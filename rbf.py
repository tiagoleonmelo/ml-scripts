import numpy as np

def phi_calc(x, c, sigma=5):
    total = 0
    for i in range(len(x)):
        total += (x[i] - c[i])**2
        
    total /= (2 * sigma**2)
    total *= -1
    
    return np.exp(total)

x1 = np.array([0, 0])
x2 = np.array([1, 0])
x3 = np.array([0, 1])
x4 = np.array([1, 1])

c1 = np.array([0, 0])
c2 = np.array([1, 0])
c3 = np.array([0, 1])
c4 = np.array([1, 1])

sigma = 1


phi1 = np.array([
        1,
        phi_calc(x1, c1, sigma),
        phi_calc(x1, c2, sigma),
        phi_calc(x1, c3, sigma),
        phi_calc(x1, c4, sigma),
    ])

phi2 = np.array([
        1,
        phi_calc(x2, c1, sigma),
        phi_calc(x2, c2, sigma),
        phi_calc(x2, c3, sigma),
        phi_calc(x2, c4, sigma),
    ])

phi3 = np.array([
        1,
        phi_calc(x3, c1, sigma),
        phi_calc(x3, c2, sigma),
        phi_calc(x3, c3, sigma),
        phi_calc(x3, c4, sigma)
    ])

phi4 = np.array([
        1,
        phi_calc(x4, c1, sigma),
        phi_calc(x4, c2, sigma),
        phi_calc(x4, c3, sigma),
        phi_calc(x4, c4, sigma)
    ])

Phi = np.array([
    phi1,
    phi2,
    phi3,
    phi4
])

print("DESIGN MATRIX\n", Phi)

targets = np.array([-1, 1, 1, -1])

w = np.array([0, -1, 1, 1, -1])

net = lambda x, w : x.dot(w)
sig = lambda net : 1 / (1 + np.exp(-net))

def sign(z):
    if z >= 0:
        return 1
    return -1

epochs = 1000
maxepoch = 2
lr = 1

for epoch in range(epochs):
    print("============ Epoch", epoch+1, "============")
    changed = False
    
    for i in range(len(Phi)):
        net_aux = net(w, Phi[i])
        print("net" + str(i+1), "=", net_aux)
        
        out = sign(net_aux)
        print("o" + str(i+1), "=", out)
        
        print("target" + str(i+1) + " - output" + str(i+1) + " =", targets[i] - out)
        w = w + lr * (targets[i] - out) * Phi[i]
        print("w_new =", w)

        if targets[i] - out != 0:
            changed = True
        
        print()

    if changed == False or epoch == maxepoch:
        print("Converged at epoch", epoch)
        break
