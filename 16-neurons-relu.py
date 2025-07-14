import numpy as np

# data
data = [(0, 1), (1, 3), (3, 5)]

# weights and biases
w1 = np.random.randn(1, 10)
b1 = np.zeros((1, 10))

w2 = np.random.randn(10, 5)
b2 = np.zeros((1, 5))

w3 = np.random.randn(5, 1)
b3 = np.zeros((1, 1))

# learning rate
lr = 0.02

# activation
def relu(x):
  return np.maximum(0, x)

# activation derivative
def relu_deriv(x):
  return (x > 0).astype(float)

# mean squared error
def mse(y_pred, y_true):
  return np.mean((y_pred - y_true) ** 2)

# mean squared error gradient
def mse_grad(y_pred, y_true):
  return 2 * (y_pred - y_true)

# learning
for epoch in range(500):
  for x_scalar, y_true_scalar in data:
    # prepare data
    x = np.array([[x_scalar]])
    y_true = np.array([[y_true_scalar]])

    # forward pass
    l1 = x @ w1 + b1  # layer 1
    a1 = relu(l1)     # layer 1 activation 

    l2 = a1 @ w2 + b2 # layer 2
    a2 = relu(l2)     # layer 2 activation

    l3 = a2 @ w3 + b3 # layer 3
    y_pred = l3       # layer 3 without activation

    # loss
    loss = mse(y_pred, y_true)

    # backpropagation
    da3 = mse_grad(y_pred, y_true)
    dl3 = da3                       # layer 3 gradient
    dw3 = a2.T @ dl3                # layer 3 weight gradient
    db3 = dl3                       # layer 3 bias gradient

    da2 = dl3 @ w3.T
    dl2 = da2 * relu_deriv(l2)      # layer 2 gradient
    dw2 = a1.T @ dl2                # layer 2 weight gradient
    db2 = dl2                       # layer 2 bias gradient

    da1 = dl2 @ w2.T
    dl1 = da1 * relu_deriv(l1)      # layer 1 gradient
    dw1 = x.T @ dl1                 # layer 1 weight gradient
    db1 = dl1                       # layer 1 bias gradient

    # update weights
    w3 -= lr * dw3
    w2 -= lr * dw2
    w1 -= lr * dw1
    b3 -= lr * db3
    b2 -= lr * db2
    b1 -= lr * db1
  
  if epoch % 10 == 0:
    print(f'epoch: {epoch}, loss: {loss}')

x_test = np.array([[2]])
l1 = x_test @ w1 + b1
a1 = relu(l1)
l2 = a1 @ w2 + b2
a2 = relu(l2)
l3 = a2 @ w3 + b3
a3 = relu(l3)
print(f'Prediction: {a3}')




