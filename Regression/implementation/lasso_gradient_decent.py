import numpy as np
import matplotlib.pyplot as plt

# l1 regularizion is also called lasso regression
 
# occurs when the the features age greater the samples
N = 50
D = 50

X =  (np.random.random((N, D)) - 0.5) *10
# gaussian distribution
true_w = np.array([1, 0.5, -0.5] + [0] * (D-3))

y = X.dot(true_w) + np.random.randn(N) * 0.5

costs = []
w = np.random.randn(D) / np.sqrt(D)
learning_rate = 0.001
l1 = 10
for i in range(500):
	yhat = X.dot(w)
	theta = yhat - y
	w = w - learning_rate * (X.T.dot(theta) + l1 * np.sign(w))

	mse = theta.dot(theta) / N
	costs.append(mse)
plt.plot(costs)
plt.show()

print("final w :", w)
plt.plot(true_w, label = "true w")
plt.plot(w, label = "w map")
plt.legend()
plt.show()