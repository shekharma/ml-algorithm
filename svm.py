import numpy as np

x= np.array([[3,2], [1,1], [2,1], [2,3]])
y = np.array([1,-1,-1,1])

lr =0.01
lambd = 0.01
epochs = 1000
n_samples, n_features = x.shape
y_ = np.where(y<=0, -1, 1)
w = np.zeros(n_features)
b = 0

for epoch in range(epochs):
    dw = 0
    db = 0
    hige_loss = 0
    for idx, x_i in enumerate(x):
        condition = y_[idx]*(np.dot(x_i, w) + b) >=1
        margin = y_[idx]*(np.dot(x_i, w)+b)
        hige_loss+= max(0, 1-margin)
        if condition:
            dw = w
            db = 0
        else:
            dw = w- lambd*y_[idx]*x_i
            db = - lambd*y_[idx]
        dw += dw
        db += db
    # averagg 
    dw = dw/n_samples
    db = db/n_samples
    hige_loss = hige_loss/n_samples
    w-=lr*dw
    b-=lr*db

    if epoch%100==0:
        print(f"Epoch {epoch}, Loss: {hige_loss:.4f}, W: {w}, b: {b:.4f}")

print("\nFinal:", w, b)