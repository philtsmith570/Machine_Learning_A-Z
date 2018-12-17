# SupportVectorReg
"""
Created on Sun Feb 18 11:24:55 2018

@author: philt
"""

# Regression Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')

# Use 1:2 to create a matrix vers 1 which creates a vector
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2:].values

# EDA - plot raw data
#plt.scatter(X, y, color = 'purple')
#plt.title('EDA - Raw Data Plot')
#plt.xlabel('Position level')
#plt.ylabel('Salary')
#plt.show()

# Splitting the dataset into the Training set and Test set
# Don't have enough data.  Want to make an accurate prediction

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

# Fitting SVR to the data set
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X, y)

# Visualizing the SVR eesults

X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))

plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')

print("SVRResult for a 6.5 Level:  ", 
      sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]])))))
