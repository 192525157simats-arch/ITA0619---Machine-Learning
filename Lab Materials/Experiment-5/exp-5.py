import numpy as np

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

np.random.seed(1)
W1 = np.random.randn(2,4)
W2 = np.random.randn(4,1)

def sigmoid(x):
    return 1/(1+np.exp(-x))

for i in range(10000):
    h = sigmoid(X @ W1)
    o = sigmoid(h @ W2)
    e = y-o

    d = e*o*(1-o)
    dh = (d @ W2.T)*h*(1-h)

    W2 += 0.5*h.T @ d
    W1 += 0.5*X.T @ dh

print("Predicted Output:")
print(np.round(o))
