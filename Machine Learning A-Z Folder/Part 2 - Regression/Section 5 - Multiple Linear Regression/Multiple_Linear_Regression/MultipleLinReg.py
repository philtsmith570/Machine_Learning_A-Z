# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 10:08:44 2018

@author: philt
"""

# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable TRAP - Remove CA column  The python libaray will
# automatically take this into account
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#y_pred = regressor.predict(X_test)

# Building the optimal model using Backward Elimination
# Need to Add a X0 to B0 for Stats model since it won't take this into account
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)

#Optimal matrix contains variables with high impact on profit
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
SL =0.05
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 1, 3, 4, 5]]
SL =0.05
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 3, 5]]
SL =0.05
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()

X_opt = X[:, [0, 3, 5]]
SL =0.05
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
print(regressor_OLS.summary())

#y_pred = regressor_OLS.predict(X_test)
#print(regressor_OLS.summary())

#Visualizing the Training set results
#fig = plt.figure()
#title = fig.suptitle('Profit Vs Model')
#fig.subplots_adjust(top=0.85, wspace=0.3)
#
#ax1 = fig.add_subplot(121)
#ax1.set_xlabel('Cost (Training Set)')
#ax1.set_ylabel('Profit')
##ax1.scatter(X_train, y_train, color= 'red')
#ax1.plot(X_opt, y_pred, color= 'blue')

##Automated BE
#import statsmodels.formula.api as sm
#def backwardElimination(x, sl):
#    numVars = len(x[0])
#    for i in range(0, numVars):
#        regressor_OLS = sm.OLS(y, x).fit()
#        maxVar = max(regressor_OLS.pvalues).astype(float)
#        if maxVar > sl:
#            for j in range(0, numVars - i):
#                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
#                    x = np.delete(x, j, 1)
#    print("No R-Squared: ", regressor_OLS.summary())
#    return x
# 
#SL = 0.05
#X_opt = X[:, [0, 1, 2, 3, 4, 5]]
#X_Modeled = backwardElimination(X_opt, SL)
#
#
##Automated BE with Adjusted R-Squared
#import statsmodels.formula.api as sm
#def backwardElimination(x, SL):
#    numVars = len(x[0])
#    temp = np.zeros((50,6)).astype(int)
#    for i in range(0, numVars):
#        regressor_OLS = sm.OLS(y, x).fit()
#        maxVar = max(regressor_OLS.pvalues).astype(float)
#        adjR_before = regressor_OLS.rsquared_adj.astype(float)
#        if maxVar > SL:
#            for j in range(0, numVars - i):
#                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
#                    temp[:,j] = x[:, j]
#                    x = np.delete(x, j, 1)
#                    tmp_regressor = sm.OLS(y, x).fit()
#                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
#                    if (adjR_before >= adjR_after):
#                        x_rollback = np.hstack((x, temp[:,[0,j]]))
#                        x_rollback = np.delete(x_rollback, j, 1)
##                        print()
##                        print("*" * 49)
##                        print("BE with R-Squared: ")
##                        print(regressor_OLS.summary())
#                        return x_rollback
#                    else:
#                        continue
#    print()
#    print("*" * 49)
#    print("BE with R-Squared: ")
#    print(regressor_OLS.summary())
#    return x
# 
#SL = 0.05
#X_opt = X[:, [0, 1, 2, 3, 4, 5]]
#X_Modeled = backwardElimination(X_opt, SL)
#print(regressor_OLS.summary())




