# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 08:23:15 2018

@author: philt
"""

# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values  # Select indep variable (All but last column)
y = dataset.iloc[:, 1].values  # Select Depedent variables (Last Column)


# Taking care of missing data
# Encoding categorical data
# Dummy encoding.  Want to decode the countries

# EDA - Visualizing the Raw Data
#plt.scatter(X, y, color= 'red')
#plt.title('Salary Vs Experiance (Raw Data Set)')
#plt.xlabel('Years of Experiance')
#plt.ylabel("Salary")
#plt.show()

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

# Feature Scaling
'''from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
# sc_y = StandardScaler
# y_train = sc_y.fit_transform(y_train)'''

#Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

#Visualizing the Training set results
fig = plt.figure()
title = fig.suptitle('Salary Vs Experiance (Training Set)')
fig.subplots_adjust(top=0.85, wspace=0.3)

ax1 = fig.add_subplot(121)
ax1.set_xlabel('Years (Training Set)')
ax1.set_ylabel('Salary')
ax1.scatter(X_train, y_train, color= 'red')
ax1.plot(X_train, regressor.predict(X_train), color= 'blue')

#Visualizing the Test set results
ax2 = fig.add_subplot(122)
ax2.set_xlabel('Years (Test Set)')
ax2.set_ylabel('Salary')
ax2.scatter(X_test, y_test, color= 'red')
ax2.plt.plot(X_train, regressor.predict(X_train), color= 'blue')

plt.show()