import numpy as np
import random

x=[1,2, 3, 7]
y = [2, 4.1, 5.9, 14.2]
W= random.uniform(1, 5)
b= random.uniform(0.5, 5)

#learning rate
lr =0.001
epochs =1000

n = len(x)

for epoch in range(epochs):
    dw=0
    db = 0
    loss =0


    for i in range(n):
        y_hat = W*x[i] + b
        error = y[i]-y_hat
        loss += error**2

        dw += -2*x[i]*error
        db += -2*error

    #averaging error
    dw = dw/n
    db = db/n

    W = W -lr*dw
    b = b -lr*db

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}, W: {W:.4f}, b: {b:.4f}")

print("\nFinal:", W, b)

print("###################################################################################")
# Let's right using numpy- all operation at once and not the pointwise

x = np.array([1,2, 3, 7])
y = np.array([2, 4.1, 5.9, 14.2])


for epoch in range(epochs):
     # vectorized prediction
    y_hat = W * x + b

    # vectorized error
    error = y - y_hat

    # vectorized loss
    loss = np.mean(error ** 2)

    # vectorized gradients
    dW = -2 * np.mean(x * error)
    db = -2 * np.mean(error)

    # update
    W -= lr * dW
    b -= lr * db

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

print("Final:", W, b)


