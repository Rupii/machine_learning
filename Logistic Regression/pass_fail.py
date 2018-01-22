import numpy as np
import pandas as pd
# importing computational dependencies

import matplotlib.pyplot as plt
# importing vizualization depenenceis

X = np.array([[70, 20], [90, 15], [60, 50], [60, 30]])
y = np.array([0, 1, 1, 0])

plt.plot(X, y, '*')
plt.show()




from sklearn.linear_model import LogisticRegression


logistic = LogisticRegression()

logistic.fit(X, y)



test = np.array([[60, 10], [100, 10], [90, 20]])

logistic.predict(test)

