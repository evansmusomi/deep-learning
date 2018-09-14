""" artificial neural networl """
# import the libraries
import numpy as numpy
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as pyplot
import pandas as pandas

# import the dataset
dataset = pandas.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# encode categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# splitting the dataset into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

# feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# make the ANN
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout


def build_classifier():
    # construct the ANN
    classifier = Sequential()
    # add the input layer, the hidden layers and output layer
    classifier.add(Dense(activation="relu", input_dim=11,
                         units=6, kernel_initializer="uniform"))
    classifier.add(Dropout(p=0.1))

    classifier.add(Dense(activation="relu", units=6,
                         kernel_initializer="uniform"))
    classifier.add(Dropout(p=0.1))
    classifier.add(Dense(activation="sigmoid", units=1,
                         kernel_initializer="uniform"))
    # compile the ANN
    classifier.compile(
        optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return classifier


from sklearn.metrics import confusion_matrix


def train_and_predict():
    classifier = build_classifier()
    # fit the ANN to the training set
    classifier.fit(X_train, y_train, batch_size=10, epochs=100)
    # predict the test set results
    y_pred = classifier.predict(X_test)
    y_pred = (y_pred > 0.5)

    # make the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(cm)


# predict a single observation
""" will the following customer leave the bank?
Geography: France
Credit score: 600
Gender: Male
Age: 40
Tenure: 3
Balance: 60000
Number of Products: 2
Has Credit Card: Yes
Is Active Member: Yes
Estimated Salary: 50000
"""


def predict_observation(classifier):
    new_prediction = classifier.predict(sc_X.transform(
        numpy.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
    new_prediction = (new_prediction > 0.5)
    print(new_prediction)


# evaluate the ann
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score


def train_with_kfold():
    classifier = KerasClassifier(
        build_fn=build_classifier, batch_size=10, epochs=100)
    accuracies = cross_val_score(
        estimator=classifier, X=X_train, y=y_train, cv=10, n_jobs=-1)
    print(accuracies)
    mean = accuracies.mean()
    print(mean)
    variance = accuracies.std()
    print(variance)


train_with_kfold()
