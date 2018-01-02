#Implementation of linear regression

#importintg libraries

import numpy as np
import matplotlib.pyplot as plt

#initializing input(X) and label values(y)
X = list()
y = list()

with open("linr_regrsion_data.csv") as fh:
	for line in fh:
		l, m = line.split(',')
		X.append(float(l))
		y.append(float(m))
# we can also do the above task easily with pandas read_csv() method


X = np.array(X)
y = np.array(y)

#plioting x, y
plt.scatter(X, y)
plt.show()

# appling the linear regession line equation

denom =  X.dot(X) -  X.mean() * X.sum() 
#since X, y are ndarray we can call the methods of numpy directly

a = ( X.dot(y) - y.mean() *X.sum() ) / denom

b = (y.mean() * X.dot(X) - X.mean() *X.dot(y) ) / denom

#predictive model

yhat = a * X + b
plt.scatter(X, y)
plt.plot(X, yhat)
plt.show()


# Error metrics are

Rres = ((y - yhat) ** 2).sum()
Rtot = ((y - y.mean()) ** 2).sum()
print(1-(Rres/Rtot))

print("Mean  Error is", (yhat - y) .mean())

print("Mean Squared Error is", ((yhat - y) ** 2).mean())