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
x = array[:,0:-1]
y = array[:,-1]
X_train, X_validation, Y_train, Y_validation = train_test_split(x, y, test_size=0.20, random_state=1)

result = []

k = 3
cv_results = cross_val_score(KNeighborsClassifier(n_neighbors = k), X_train, Y_train, scoring='accuracy')
print("k = 3")
print('%s: %f (%f)' % ("KNN", cv_results.mean(), cv_results.std()))

k = 5
cv_results = cross_val_score(KNeighborsClassifier(n_neighbors = k), X_train, Y_train, scoring='accuracy')
print("k = 5")
print('%s: %f (%f)' % ("KNN", cv_results.mean(), cv_results.std()))

k = 11
cv_results = cross_val_score(KNeighborsClassifier(n_neighbors = k), X_train, Y_train, scoring='accuracy')
print("k = 11")
print('%s: %f (%f)' % ("KNN", cv_results.mean(), cv_results.std()))

k = 17
cv_results = cross_val_score(KNeighborsClassifier(n_neighbors = k), X_train, Y_train, scoring='accuracy')
print("k = 17")
print('%s: %f (%f)' % ("KNN", cv_results.mean(), cv_results.std()))

#Linear Regression

dummies = get_dummies(dataset,drop_first=True)

array = dummies.values
x = array[:,0:-1]
y = array[:,-1]
X_train, X_validation, Y_train, Y_validation = train_test_split(x, y, test_size=0.20, random_state=1)

cv_results = cross_val_score(LogisticRegression(penalty="l2"), X_train, Y_train, scoring='accuracy')
print("penalty = l2")
print('%s: %f (%f)' % ("LR", cv_results.mean(), cv_results.std()))

cv_results = cross_val_score(LogisticRegression(penalty=None), X_train, Y_train, scoring='accuracy')
print("penalty = None")
print('%s: %f (%f)' % ("LR", cv_results.mean(), cv_results.std()))


#DecisionTreeClassifier
dataset = read_csv(path)
dummies = get_dummies(dataset, dtype = int,drop_first=True)

array = dummies.values
x = array[:,0:-1]
y = array[:,-1]
X_train, X_validation, Y_train, Y_validation = train_test_split(x, y, test_size=0.20, random_state=1)

cv_results = cross_val_score(DecisionTreeClassifier(criterion="gini"), X_train, Y_train, scoring='accuracy')
cv_results = cross_val_score(DecisionTreeClassifier(criterion="gini"), X_validation, Y_validation, scoring='accuracy')
print("gini")
print('%s: %f (%f)' % ("DTC", cv_results.mean(), cv_results.std()))

cv_results = cross_val_score(DecisionTreeClassifier(criterion="entropy"), X_train, Y_train, scoring='accuracy')
print("entropy")
print('%s: %f (%f)' % ("DTC", cv_results.mean(), cv_results.std()))





def train_and_evaluate(max_depth=None):
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    clf.fit(X_train, y_train)
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    
    train_recall = recall_score(y_train, y_pred_train)
    test_recall = recall_score(y_test, y_pred_test)
    
    return train_accuracy, test_accuracy, train_recall, test_recall, clf
