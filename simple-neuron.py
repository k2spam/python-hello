def simple_neuron(x1, x2, x3, w1, w2, w3): 
  return x1 * w1 + x2 * w2 + x3 * w3

def activation(x):
  return 1 if x >= 1 else x 

output = activation(simple_neuron(4, 7, 9, 0.5, 0.9, 0.9))
print(output)