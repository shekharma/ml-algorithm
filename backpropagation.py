import numpy as np

# ---- Data (simple binary classification) ----
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

Y = np.array([[0], [1], [1], [0]])  # XOR problem

# ---- Initialize weights ----
np.random.seed(0)

W1 = np.random.randn(2, 2) * 0.1
b1 = np.zeros((1, 2))

W2 = np.random.randn(2, 1)
b2 = np.zeros((1, 1))

lr = 0.1

# ---- Activation functions ----
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

# ---- Training ----
for epoch in range(10000):

    # ===== FORWARD PASS =====
    Z1 = X @ W1 + b1
    A1 = relu(Z1)

    Z2 = A1 @ W2 + b2
    A2 = sigmoid(Z2)   # prediction

    # ===== LOSS =====
    loss = np.mean((A2 - Y) ** 2)

    # ===== BACKWARD PASS =====

    # Output layer
    dA2 = 2 * (A2 - Y)
    dZ2 = dA2 * sigmoid_derivative(Z2)

    dW2 = A1.T @ dZ2          # <-- key line
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    # Hidden layer
    dA1 = dZ2 @ W2.T          # <-- W2 appears here
    dZ1 = dA1 * relu_derivative(Z1)

    dW1 = X.T @ dZ1           # <-- key line
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    # ===== UPDATE =====
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")
        
print(f" final weights w1-{W1}, w2-{W2}, b1- {b1}, b2 - {b2}")

# ---- Final Output ----
print("\nPredictions:")
print(A2)