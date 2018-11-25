# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 09:31:17 2018

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
y = dataset.iloc[:, 2].values

# EDA - plot raw data
#plt.scatter(X, y, color = 'purple')
#plt.title('EDA - Raw Data Plot')
#plt.xlabel('Position level')
#plt.ylabel('Salary')
#plt.show()

# Splitting the dataset into the Training set and Test set
# Don't have enough data.  Want to make an accurate prediction

# Feature Scaling
# No need here - Library habdles this

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
LinReg = LinearRegression()
LinReg.fit(X, y)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
PolyReg = PolynomialFeatures(degree=4)  # Start with degree = 2
X_poly = PolyReg.fit_transform(X)
PolyLinReg = LinearRegression()
PolyLinReg.fit(X_poly, y)

# Plot residuals
''' Linear Residual plot shows a U shape suggesting a better fit
using a non-linear model'''
res_lin  = y - LinReg.predict(X)
res_poly = y - PolyLinReg.predict(PolyReg.fit_transform(X))
plt.figure(figsize=(12,8), facecolor='1.0')
plt.scatter(X, res_lin)
plt.title('EDA - Residual Data Plot (Simple Linear)', size=28)
plt.xlabel('Position level', size=24)
plt.ylabel('y-yhat', size=24)
plt.show()

''' Polynomial Residual plot shows a random pattern'''
plt.figure(figsize=(12,8), facecolor='1.0')
plt.scatter(X, res_poly)
plt.title('EDA - Residual Data Plot (Polynimial)', size=28)
plt.xlabel('Position level', size=24)
plt.ylabel('y-yhat', size=24)
plt.show()

# Check Q-Q plot (Normally Distributed)
import numpy.random as random
'''Plot shows the one outlier, but besides that has a
normal distribution'''

#y_test = y[0:9] # Removed outlier for testing
y_test.sort()
norm = random.normal(0, 2, len(y))
norm.sort()
plt.figure(figsize=(12,8), facecolor='1.0')
plt.plot(norm, y, "o")

#Generate a trend line
z = np.polyfit(norm, y, 1)
p = np.poly1d(z)
plt.plot(norm, p(norm), "--", linewidth=2)
plt.title("Normal Q-Q Plot", size = 28)
plt.xlabel("Theoretical Quantiles", size=24)
plt.ylabel("Salary Quantiles", size=24)
plt.tick_params(labelsize=16)
plt.show()

## Visualizing the Linear Regression
#plt.scatter(X, y, color = 'red')
#plt.plot(X, LinReg.predict(X), color='blue')
#plt.title('Truth or Bluff (Linear Regression)')
#plt.xlabel('Position level')
#plt.ylabel('Salary')
#
## Visualizing the Poly Regression
#X_grid = np.arange(min(X), max(X), 0.1)
#X_grid = X_grid.reshape((len(X_grid), 1))
#plt.scatter(X, y, color = 'red')
#plt.plot(X_grid, PolyLinReg.predict(PolyReg.fit_transform(X_grid)), 
#         color='green')
#plt.title('Truth or Bluff (Polynomial Regression)')
#plt.xlabel('Position level')
#plt.ylabel('Salary')
#plt.show()

# Predictin a new result with Linear Regression
print("Linear Regression Result for a 6.5 Level:  ", LinReg.predict(6.5))

# Predicting a new result with Polynimial Regression
print("Poly Regression Result for a 6.5 Level:  ", 
      PolyLinReg.predict(PolyReg.fit_transform(6.5)))
