import numpy as np

EPOCHS = 100000
lr = 0.1
TARGET_MSE = 0.01

X = np.array([[0,0], [0,1], [1,1], [1,0]])
Y = np.array([0, 1, 0, 1])

layers = [2, 2, 1] 

np.random.seed(42)
W = [np.random.randn(layers[i], layers[i+1]) * 0.1 for i in range(len(layers)-1)]
B = [np.random.randn(layers[i+1]) * 0.1 for i in range(len(layers)-1)]

def sigma(z):
    z = np.clip(z, -500, 500) 
    return 1 / (1 + np.exp(-z))

def dsigma(z):
    s = sigma(z)
    return s * (1 - s)

def forward(z, W, B, z_list):
    a = z
    for index, wn in enumerate(W):
        z = a @ wn + B[index]
        z_list[index] = z
        a = sigma(z)
    return a, z_list

def backward(y, y_hat, z_list, x, W):
    L = len(W)
    dEdW = [None] * L
    dEdB = [None] * L
    
    delta = (y_hat - y) * dsigma(z_list[-1])
    
    for l in reversed(range(L)):
        if l == 0:
            a_prev = x
        else:
            a_prev = sigma(z_list[l-1])
            
        dEdW[l] = a_prev.reshape(-1, 1) @ delta.reshape(1, -1)
        dEdB[l] = delta
        
        if l > 0:
            delta = delta @ W[l].T * dsigma(z_list[l-1])
            
    return dEdW, dEdB

def train(X, Y, W, B, EPOCHS, lr, target_mse):
    z_list = [None] * len(W)
    
    mse_history = []
    weights_history = []
    classification_error_history = []
    
    for epoch in range(EPOCHS):
        
        epoch_mse = 0
        correct_predictions = 0
        
        for x, y in zip(X, Y):
            y_hat, z_list = forward(x, W, B, z_list)
            
            sample_error = 0.5 * (y_hat - y)**2
            epoch_mse += sample_error.item()
            
            dEdW, dEdB = backward(y, y_hat, z_list, x, W)
            
            for i in range(len(W)):
                W[i] -= lr * dEdW[i]
                B[i] -= lr * dEdB[i]

            y_predicted_class = 1 if y_hat.item() >= 0.5 else 0
            if y_predicted_class == y:
                correct_predictions += 1
                
        avg_mse = epoch_mse / len(X)
        mse_history.append(avg_mse)
        
        classification_error = 1.0 - (correct_predictions / len(X))
        classification_error_history.append(classification_error)
        
        flat_weights = np.concatenate([w.flatten() for w in W])
        weights_history.append(flat_weights)
        
        if avg_mse <= target_mse:
            print(f"Training finished after epoch {epoch + 1}. Achieved MSE: {avg_mse:.6f} <= {target_mse}")
            break
            
        if (epoch + 1) % 10000 == 0:
            print(f"Epoch: {epoch + 1}, MSE: {avg_mse:.6f}, Classification Error: {classification_error:.2f}")

    return W, B, mse_history, classification_error_history, weights_history

W, B, mse_history, classification_error_history, weights_history = train(X, Y, W, B, EPOCHS, lr, TARGET_MSE)

def XOR(z, W, B):
    y_hat, _ = forward(z, W, B, [None]*len(W))
    return 1 if y_hat.item() >= 0.5 else 0

for x in X:
    result = XOR(x, W, B)
    print(f"Input: {x} -> Result: {result}")