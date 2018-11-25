# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset  - use tab delimiter and quoting to ignore quotes
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 1000):
    # Keep only letters
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    # Lowercase
    review = review.lower()
    # Split into individual words
    review = review.split()
    # Define stemmer object
    ps = PorterStemmer()
    
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

Accuracy = (cm[0][0] + cm[1][1]) / (cm[0][1] + cm[1][0] + cm[0][0] + cm[1][1])
Precision = (cm[0][0]) / (cm[0][0] + cm[0][1])
Recall = (cm[0][0]) / (cm[0][0] + cm[1][0])
F1_Score = 2 * (Precision*Recall/(Precision + Recall))

print('Accuracy: ', Accuracy)
print('Precision: ', Precision)
print('Recall: ', Recall)
print('F1 Score: ', F1_Score)

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifierRF = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifierRF.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifierRF.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cmRF = confusion_matrix(y_test, y_pred)

cmRF = confusion_matrix(y_test, y_pred)

AccuracyRF = (cmRF[0][0] + cmRF[1][1]) / (cmRF[0][1] + cmRF[1][0] + cmRF[0][0] + cmRF[1][1])
PrecisionRF = (cmRF[0][0]) / (cmRF[0][0] + cmRF[0][1])
RecallRF = (cmRF[0][0]) / (cmRF[0][0] + cmRF[1][0])
F1_ScoreRF = 2 * (PrecisionRF*RecallRF/(PrecisionRF + RecallRF))

print('Accuracy: ', AccuracyRF)
print('Precision: ', PrecisionRF)
print('Recall: ', RecallRF)
print('F1 Score: ', F1_ScoreRF)