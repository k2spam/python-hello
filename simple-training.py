data = [(1, 2), (2, 4), (3, 6)] # data [(x, y_true),...] rule: y=x*2

w = 0.5   # start weight
lr = 0.01 # learning speed

def neuron(x, w):
  return x * w

for epoch in range(100):
  for x, y_true in data:
    y_pred = neuron(x, w) # prediction
    loss = (y_pred - y_true) ** 2 # error (MSE)
    gradient = 2 * (y_pred - y_true) * x 
    w -= lr * gradient # weight correction

input = 7
output = 7 * 2
print(f'Weight: {w}\n prediction:\n neuron output {neuron(7, w)}\nshould be: {output}')