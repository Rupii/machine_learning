# basic logistic regreesion code
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x = np.random.randn(50, 2)
ones = np.array([[1]*50]).T
X = np.concatenate((ones, x), axis = 1)

theta = np.random.randn(3)
yhat = X.dot(theta)
sigmoid =  1 / (1 + np.exp(-yhat))
sigmoid = np.array(sigmoid)
print(sigmoid) 


sigmoid = pd.DataFrame(sigmoid)
print(sigmoid)
def logistic(n):
	if n <= 0.5:
		return 0

	else:
		return 1
sigmoid[0]= sigmoid[0].apply(logistic)

plt.plot(sigmoid[0])
plt.show()
