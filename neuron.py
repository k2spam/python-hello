def simple_neuron(x1, x2, w1, w2):
  return x1 * w1 + x2 * w2

def activation(x):
  return 1 if x >= 1 else 0

output = activation(simple_neuron(1.5, 3, 0.5, 0.3))
print(output)

def neuron(x1, x2, x3, w1, w2, w3):
  return x1 * w1 + x2 * w2 + x3 * w3

def relu(x):
  return x if x > 0 else 0

output = relu(neuron(1, 4, 7, 0.5, 0.9, 0.7))
print(output)