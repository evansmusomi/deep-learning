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

# handle missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
print(X)
