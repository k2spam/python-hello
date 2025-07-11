def neuron(x1, x2, x3, w1, w2, w3):
  return x1 * w1 + x2 * w2 + x3 * w3

def relu(x):
  return x if x > 0 else 0

output = relu(neuron(1, 4, 7, 0.5, 0.9, 0.7))
print(output)