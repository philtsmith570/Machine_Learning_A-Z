# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values  # Select indep variable (All but last column)
y = dataset.iloc[:, 3].values  # Select Depedent variables (Last Column)

print('y', y)
print('*' * 40)
print('X raw', X)


# Taking care of missing data
from sklearn.preprocessing import Imputer
# Imputer will use the mean strategy
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
# Fit the imputer (Mean) to X
imputer = imputer.fit(X[:, 1:3])
# Impute all missing values of X ('NAN' here)
X[:, 1:3] = imputer.transform(X[:, 1:3])
print('*' * 40)
print('X after imputing', X)

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_X = LabelEncoder()
X[:, 0] = labelEncoder_X.fit_transform(X[:, 0])  # Country
labelencoder_y = LabelEncoder()
y = label_y = labelencoder_y.fit_transform(y)


# Dummy encoding.  Want to decode the countries
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
# sc_y = StandardScaler
# y_train = sc_y.fit_transform(y_train)
