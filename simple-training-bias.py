data = [(1, 3), (2, 5), (3, 7)] # [(x, y_true), ...] rule: y = 2*x + 1

w, b = 0.5, 0.1 # weight and bias
lr = 0.01 # learning rate

def neuron(x, w, b):
  return x * w + b

for epoch in range(1000):
  for (x, y_true) in data:
    y_pred = neuron(x, y_true, b) # prediction
    
    grad_w = 2 * (y_pred - y_true) * x
    grad_b = 2 * (y_pred - y_true)

    w -= lr * grad_w
    b -= lr * grad_w

input = 7
output = 7 * 2 + 1
print(f'output weight: {w}\noutput bias: {b}\nneuron prediction: {neuron(input, w, b)}\nshould be: {output}')

