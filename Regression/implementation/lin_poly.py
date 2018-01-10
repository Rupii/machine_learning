import numpy as np
import matplotlib.pyplot as plt

X = []
y = []
#opening the csv file
for line in open("poly_data.csv"):
	x, p = line.split(",")
	x = float(x)
	X.append([1, x, x ** 2])
	y.append(float(p))

X = np.array(X)
y = np.array(y)
theta = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, y))
yhat = X.dot(theta)
plt.scatter(X[:, 1], y)

# if we dont sort the plot chooses random points and draws a line between which looks wired
# a quadratic function is a motonically increasing funtion
# doest't matter if it is sorted or not
plt.plot(sorted(X[:, 1]), sorted(yhat))
plt.show()

d1 = y - yhat
d2 = y - yhat.mean()

Rsquared = 1 - (d1.dot(d1)/d2.dot(d2))
print(Rsquared)

# if the rsquared is approx to 1 it is considered a good fit
#if it is negative that would be worst fit