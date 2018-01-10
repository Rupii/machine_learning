from __future__ import print_function, division
from builtins import range

import numpy as np
import matplotlib.pyplot as plt

# l2 regularization is also called ridge regression

N = 50

# generate the data
X = np.linspace(0,10,N)
y = 0.5*X + np.random.randn(N)

# making outliers
# so that these can diverse the best fit
y[-1] += 30
y[-2] += 30

# plot the data
plt.scatter(X, y)
plt.show()

# add bias term
X = np.vstack([np.ones(N), X]).T

# plot the maximum likelihood solution
w_ml = np.linalg.solve(X.T.dot(X), X.T.dot(y))
Yhat_ml = X.dot(w_ml)
plt.scatter(X[:,1], y)
plt.plot(X[:,1], Yhat_ml)
plt.show()

# plot the regularized solution
# probably don't need an L2 regularization this high in many problems
# everything in this example is exaggerated for visualization purposes
l2 = 1000.0
w_map = np.linalg.solve(l2*np.eye(2) + X.T.dot(X), X.T.dot(y))
Yhat_map = X.dot(w_map)
plt.scatter(X[:,1], y)
plt.plot(X[:,1], Yhat_ml, label='maximum likelihood')
plt.plot(X[:,1], Yhat_map, label='map')
plt.legend()
plt.show()

#you can run for multipe random l2 to choose the best fit