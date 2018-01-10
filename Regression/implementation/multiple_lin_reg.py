# importing libraries
import numpy as np
import pandas as pd


#we can open file but lets go throug data frames htis time
df = pd.read_csv("data_2d.csv", header = None)

# pandas converts internally all the colomns into nd array
#no need of converting to np objects once again
X = df[[0, 1]]
y = df[2]
X[2] = 1
# since there are thre features 
#our equation wolud be yhat = m1x1 + m2x2 +m2x3
# x1 = 1 as it would be easy for matrix mulication
theta = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, y))

#theta gives us m1, m2, m3

yhat = np.dot(X, theta)


d1 = y - yhat 
d2 = y - yhat.mean()
Rsquared = 1 - (d1.dot(d1)/d2.dot(d2))
print(Rsquared)