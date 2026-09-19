import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 1. Loss class: Mean Squared Error (MSE)
class MSELoss:
    def __call__(self, y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)

    def gradient(self, y_pred, y_true, X):
        m = len(y_true)
        return (2 / m) * X.T.dot(y_pred - y_true)

# 2. Optimizer class: Gradient Descent
class GradientDescent:
    def __init__(self, lr=0.1, n_iter=1000):
        self.lr = lr
        self.n_iter = n_iter
        self.loss_history = []

    def optimize(self, X, y, theta, loss_fn):
        for i in range(self.n_iter):
            y_pred = X.dot(theta)
            loss = loss_fn(y_pred, y)
            self.loss_history.append(loss)
            grad = loss_fn.gradient(y_pred, y, X)
            theta = theta - self.lr * grad
        return theta


# 3. Data: UCI Wine Quality Red
data = pd.read_csv(r"C:\Users\Lenovo\Downloads\wine+quality\winequality-red.csv", sep=";")

X = data[["alcohol"]].values
y = data[["quality"]].values.astype(float)

X = (X - X.mean(axis=0)) / X.std(axis=0)
X_b = np.c_[np.ones((len(X), 1)), X]

# 4. Training
theta = np.random.randn(2, 1)
loss_fn = MSELoss()
optimizer = GradientDescent(lr=0.1, n_iter=1000)
theta = optimizer.optimize(X_b, y, theta, loss_fn)

print("Learned parameters:", theta.ravel())

# 5. Plot
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(optimizer.loss_history)
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Loss Curve")

plt.subplot(1, 2, 2)
plt.scatter(X, y, label="Original data")
plt.plot(X, X_b.dot(theta), color="red", label="Fitted line")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Linear Regression Fit")

plt.tight_layout()
plt.show()