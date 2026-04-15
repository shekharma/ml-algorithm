# logistic regression
# sigmoid function = y_hat= f(x) = 1/(1+e^-z)  and z = -wx+b
# log-loss = -(y*log(y_hat)+ (1-y)*log(1-y_hat))
# dl/dw = (y_hat - y)*x
# dl/db = (y_hat - y)

import random
import math

w = random.uniform(-1, 1)
b = random.uniform(-1, 1)

x = [0, 3, 7, 8, 9]
y = [0, 0, 1, 1, 1]

lr = 0.01
epochs = 1000
n = len(x)

epsilon = 1e-9 ## addition of this is necessary beacuse log(0) is not defined y_hat =0 or y_hat=1

for epoch in range(epochs):
    dw = 0
    db = 0
    total_loss = 0

    for i in range(n):
        z = w * x[i] + b
        y_hat = 1 / (1 + math.exp(-z))

        # clipping
        y_hat = max(epsilon, min(1 - epsilon, y_hat))

        loss = -(y[i]*math.log(y_hat) + (1-y[i])*math.log(1-y_hat))
        total_loss += loss

        dw += (y_hat - y[i]) * x[i]
        db += (y_hat - y[i])

    # average
    dw /= n
    db /= n
    total_loss /= n

    # update
    w -= lr * dw
    b -= lr * db

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss:.4f}, W: {w:.4f}, b: {b:.4f}")

print("\nFinal:", w, b)

print("###################################################################################")
# Let's right using numpy- all operation at once and not the pointwise
import numpy as np
x = np.array([0, 3, 7, 8, 9])
y = np.array([0, 0, 1, 1, 1])


lr = 0.01
epochs = 1000
n = len(x)

epsilon = 1e-9

for epoch in range(epochs):
    dw = 0
    db = 0
    total_loss = 0

    z = w*x + b
    y_hat = 1/ (1+np.exp(-z))
    y_hat = np.clip(epsilon,epsilon, 1 - epsilon)

    loss = np.mean(-(y*np.log(y_hat)+ (1-y)*np.log(1-y_hat)))

    dw = np.mean((y_hat - y)*x)
    db = np.mean(y_hat - y)
    # update
    w -= lr * dw
    b -= lr * db

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

print("Final:", w, b)