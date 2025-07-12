import numpy as np

data = [(0, 1), (1, 3), (3, 5)]

w1, w2, w3 = 0.5, 0.6, 0.3
b1, b2, b3 = 0.1, 0.2, 0.3
lr = 0.000903

def relu(x):
  return max(0, x)

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def neuron(x, w, b):
  return x * w + b

for epoch in range(1000):
  for x, y_true in data:
    # forward pass
    l1_input = neuron(x, w1, b1)
    l1 = sigmoid(l1_input)

    l2_input = neuron(l1, w2, b2)
    l2 = sigmoid(l2_input) # layer 2

    l3_input = neuron(l2, w3, b3)
    l3 = sigmoid(l3_input) # prediction

    # backpropаgation
    # output layer 3
    d_l3_input = l3 * (1 - l3) # derivative
    grad_l3_input = 2 * (l3 - y_true) * d_l3_input
    grad_w3 = grad_l3_input * l2
    grad_b3 = grad_l3_input

    # hidden layer 2
    d_l2_input = l2 * (1 - l2)
    grad_l2_input = grad_l3_input * w3 * d_l2_input
    grad_w2 = grad_l2_input * l1
    grad_b2 = grad_l2_input

    # input layer 1
    d_l1_input = l1 * (1 - l1)
    grad_l1_input = grad_l2_input * w2 * d_l1_input
    grad_w1 = grad_l1_input * x
    grad_b1 = grad_l1_input

    #update weight and bias
    w1 -= lr * grad_w1
    b1 -= lr * grad_b1
    w2 -= lr * grad_w2
    b2 -= lr * grad_b2
    w3 -= lr * grad_w3
    b3 -= lr * grad_b3

input = 1
test_layer1 = neuron(input, w1, b1)
test_layer2 = neuron(test_layer1, w2, b2)
test_layer3 = neuron(test_layer2, w3, b3)
print(f'Prediction: {test_layer3}')