""" Data Pre-Processing """
import numpy as numpy
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as pyplot
import pandas as pandas

# import the dataset
dataset = pandas.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# splitting the dataset into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

# feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
