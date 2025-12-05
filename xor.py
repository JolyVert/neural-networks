import numpy as np

EPOCHS = 100000
lr = 0.01

X = np.array([[0,0], [0,1], [1,1], [1,0]])
Y = np.array([0, 1, 0, 1])

layers = [2, 2, 1] 

W = [np.random.randn(layers[i], layers[i+1]) for i in range(len(layers)-1)]
B = [np.random.randn(layers[i+1]) for i in range(len(layers)-1)]

def sigma(z):
    return 1 / (1 + np.exp(-z))

def dsigma(z):
    s = sigma(z)
    return s*(1-s)

def forward(z, W, B, z_list):
    for index, wn in enumerate(W):
        z = z @ wn + B[index]
        z_list[index] = z
        z = sigma(z)
    return z, z_list


def backward(y, y_hat, z_list, x, W):
    L = len(W)
    dEdW = [None]*L
    dEdB = [None]*L
    delta = (y_hat - y) * dsigma(z_list[-1])
    for l in reversed(range(L)):
        if l == 0:
            a_prev = x
        else:
            a_prev = sigma(z_list[l-1])
        dEdW[l] = a_prev.reshape(-1,1) @ delta.reshape(1,-1)
        dEdB[l] = delta
        if l > 0:
            delta = delta @ W[l].T * dsigma(z_list[l-1])
    return dEdW, dEdB

def train(X, Y, W, B, EPOCHS, lr):
    z_list = [None]*len(W)
    for epoch in range(EPOCHS):
        for x, y in zip(X, Y):
            y_hat, z_list = forward(x, W, B, z_list)
            dEdW, dEdB = backward(y, y_hat, z_list, x, W)
            for i in range(len(W)):
                W[i] -= lr * dEdW[i]
                B[i] -= lr * dEdB[i]

train(X, Y, W, B, EPOCHS, lr)

def XOR(z, W, B):
    for index, wn in enumerate(W):
        z = z @ wn + B[index]
        z = sigma(z)
    return round(z.item())

print("Result: ")
for x in X:
    result = XOR(x, W, B)
    print(f"{x} -> {result}")
