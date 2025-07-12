import numpy as np

data = [(1, 4), (2, 7), (3, 10), (-2, -5)] # (x, y_true) rule: 3x + 1

w1, b1 = 0.5, 0.1
w2, b2 = 0.3, 0.2
lr = 0.007185

def neuron(x, w, b):
  return x * w + b

def relu(x):
  return max(0, x)

for epoch in range(10000):
  for x, y_true in data:
    # forward pass
    layer1 = relu(neuron(x, w1, b1))
    y_pred = relu(neuron(layer1, w2, b2))

    #backpropogation
    grad_y_pred = 2 * (y_pred - y_true)
    grad_w2 = grad_y_pred * layer1 if ( w2 * layer1 + b1 ) > 0 else 0
    grad_b2 = grad_y_pred if ( w2 * layer1 + b1 ) > 0 else 0

    grad_layer1 = 2 * grad_y_pred * w2 if (w2 * layer1 + b2) > 0 else 0
    grad_w1 = grad_layer1 * x if (w1 * x + b1) > 0 else 0
    grad_b1 = grad_layer1 if (w1 * x + b1) > 0 else 0

    #update weight and bias
    w1 -= lr * grad_w1
    b2 -= lr * grad_b1
    w2 -= lr * grad_w2
    b2 -= lr * grad_b2

input = 4
output = 4 * 3 + 1
prediction_layer1 = relu(neuron(input, w1, b1))
prediction_layer2 = relu(neuron(prediction_layer1, w2, b2))

print(f'prediction: {prediction_layer2}\nshould be: {output}')