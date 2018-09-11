""" Logistical Regression """
import numpy as numpy
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as pyplot
import pandas as pandas

# import the dataset
dataset = pandas.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# splitting the dataset into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0)

# feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# fit the classifer to the training set
# Create your classifier here

# predict the test set results
y_pred = classifier.predict(X_test)

# make the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# visualize the training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = numpy.meshgrid(numpy.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                        numpy.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
pyplot.contourf(X1, X2, classifier.predict(numpy.array([X1.ravel(), X2.ravel()]).T).reshape(
    X1.shape), alpha=0.75, cmap=ListedColormap(("red", "green")))
pyplot.xlim(X1.min(), X1.max())
pyplot.ylim(X2.min(), X2.max())
for i, j in enumerate(numpy.unique(y_set)):
    pyplot.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                   c=ListedColormap(("red", "green"))(i), label=j)
pyplot.title("Logistic regression (training set)")
pyplot.xlabel("Age")
pyplot.ylabel("Estimated Salary")
pyplot.legend()
pyplot.show()

# visualize the test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = numpy.meshgrid(numpy.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                        numpy.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))
pyplot.contourf(X1, X2, classifier.predict(numpy.array([X1.ravel(), X2.ravel()]).T).reshape(
    X1.shape), alpha=0.75, cmap=ListedColormap(("red", "green")))
pyplot.xlim(X1.min(), X1.max())
pyplot.ylim(X2.min(), X2.max())
for i, j in enumerate(numpy.unique(y_set)):
    pyplot.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                   c=ListedColormap(("red", "green"))(i), label=j)
pyplot.title("Logistic regression (test set)")
pyplot.xlabel("Age")
pyplot.ylabel("Estimated Salary")
pyplot.legend()
pyplot.show()
