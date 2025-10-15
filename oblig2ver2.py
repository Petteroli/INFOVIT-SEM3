import pandas as pd
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt
from pandas import get_dummies
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import metrics


path = "INFOVIT-SEM3/INFO180/Oblig2/party_data.csv"
dataset = read_csv(path)

dummies = get_dummies(dataset, dtype = int)

array = dummies.values
X = array[:,0:-1]
y = array[:,-1]
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.20, random_state=1)


k = 3
knn = KNeighborsClassifier(n_neighbors = k)
knn.fit(X_train, Y_train)
y_pred_train = knn.predict(X_train)
y_pred_test = knn.predict(X_test)
    
train_accuracy = accuracy_score(Y_train, y_pred_train)
test_accuracy = accuracy_score(Y_test, y_pred_test)
overfitting = train_accuracy - test_accuracy    

print("k = 3")
print(f'train_accuracy: {train_accuracy}')
print(f'test_accuract: {test_accuracy}')
print(f'overfitting: {overfitting}')
print("--------------------------------------------------")

k = 5
knn = KNeighborsClassifier(n_neighbors = k)
knn.fit(X_train, Y_train)
y_pred_train = knn.predict(X_train)
y_pred_test = knn.predict(X_test)
    
train_accuracy = accuracy_score(Y_train, y_pred_train)
test_accuracy = accuracy_score(Y_test, y_pred_test)
overfitting = train_accuracy - test_accuracy    

print("k = 5")
print(f'train_accuracy: {train_accuracy}')
print(f'test_accuract: {test_accuracy}')
print(f'overfitting: {overfitting}')
print("--------------------------------------------------")

k = 11
knn = KNeighborsClassifier(n_neighbors = k)
knn.fit(X_train, Y_train)
y_pred_train = knn.predict(X_train)
y_pred_test = knn.predict(X_test)
    
train_accuracy = accuracy_score(Y_train, y_pred_train)
test_accuracy = accuracy_score(Y_test, y_pred_test)
overfitting = train_accuracy - test_accuracy    

print("k = 11")
print(f'train_accuracy: {train_accuracy}')
print(f'test_accuract: {test_accuracy}')
print(f'overfitting: {overfitting}')
print("--------------------------------------------------")

k = 17
knn = KNeighborsClassifier(n_neighbors = k)
knn.fit(X_train, Y_train)
y_pred_train = knn.predict(X_train)
y_pred_test = knn.predict(X_test)
    
train_accuracy = accuracy_score(Y_train, y_pred_train)
test_accuracy = accuracy_score(Y_test, y_pred_test)
overfitting = train_accuracy - test_accuracy    

print("k = 17")
print(f'train_accuracy: {train_accuracy}')
print(f'test_accuract: {test_accuracy}')
print(f'overfitting: {overfitting}')
print("--------------------------------------------------")

#Linear Regression
dummies = get_dummies(dataset, dtype = int, drop_first=True)

array = dummies.values
X = array[:,0:-1]
y = array[:,-1]
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.20, random_state=1)

lr = LogisticRegression(penalty="l2")
lr.fit(X_train, Y_train)
y_pred_train = lr.predict(X_train)
y_pred_test = lr.predict(X_test)
    
train_accuracy = accuracy_score(Y_train, y_pred_train)
test_accuracy = accuracy_score(Y_test, y_pred_test)
overfitting = train_accuracy - test_accuracy    

print("penalty = l2")
print(f'train_accuracy: {train_accuracy}')
print(f'test_accuract: {test_accuracy}')
print(f'overfitting: {overfitting}')
print("--------------------------------------------------")

lr = LogisticRegression(penalty=None)
lr.fit(X_train, Y_train)
y_pred_train = lr.predict(X_train)
y_pred_test = lr.predict(X_test)
    
train_accuracy = accuracy_score(Y_train, y_pred_train)
test_accuracy = accuracy_score(Y_test, y_pred_test)
overfitting = train_accuracy - test_accuracy    

print("penalty = None")
print(f'train_accuracy: {train_accuracy}')
print(f'test_accuract: {test_accuracy}')
print(f'overfitting: {overfitting}')
print("--------------------------------------------------")

#DecitionTreeClassifier
dummies = get_dummies(dataset, dtype = int, drop_first=True)

array = dummies.values
X = array[:,0:-1]
y = array[:,-1]
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.20, random_state=1)

dtc = DecisionTreeClassifier(criterion="gini")
dtc.fit(X_train, Y_train)
y_pred_train = dtc.predict(X_train)
y_pred_test = dtc.predict(X_test)
    
train_accuracy = accuracy_score(Y_train, y_pred_train)
test_accuracy = accuracy_score(Y_test, y_pred_test)
overfitting = train_accuracy - test_accuracy    

print("criterion=gini")
print(f'train_accuracy: {train_accuracy}')
print(f'test_accuract: {test_accuracy}')
print(f'overfitting: {overfitting}')
print("--------------------------------------------------")

dtc = DecisionTreeClassifier(criterion="entropy")
dtc.fit(X_train, Y_train)
y_pred_train = dtc.predict(X_train)
y_pred_test = dtc.predict(X_test)
    
train_accuracy = accuracy_score(Y_train, y_pred_train)
test_accuracy = accuracy_score(Y_test, y_pred_test)
overfitting = train_accuracy - test_accuracy    

print("criterion=entropy")
print(f'train_accuracy: {train_accuracy}')
print(f'test_accuract: {test_accuracy}')
print(f'overfitting: {overfitting}')
print("--------------------------------------------------")







