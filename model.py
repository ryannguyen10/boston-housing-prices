import numpy as np
import tensorflow as tf
import pandas as pd
import os 
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from pandas import read_csv
from tensorflow import keras
import random
import math
import seaborn as sns
import sklearn.model_selection
import sklearn.linear_model
import sklearn.ensemble

#CRIM - per capita crime rate by town
#ZN - proportion of residential land zoned for lots over 25,000 sq.ft.
#INDUS - proportion of non-retail business acres per town.
#CHAS - Charles River dummy variable (1 if tract bounds river; 0 otherwise)
#NOX - nitric oxides concentration (parts per 10 million)
#RM - average number of rooms per dwelling
#AGE - proportion of owner-occupied units built prior to 1940
#DIS - weighted distances to five Boston employment centres
#RAD - index of accessibility to radial highways
#TAX - full-value property-tax rate per $10,000
#PTRATIO - pupil-teacher ratio by town
#B - 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
#LSTAT - % lower status of the population
#MEDV - Median value of owner-occupied homes in $1000's

column_names = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PIRATIO','B','LSTAT','MEDV']
housing = read_csv('housing.csv',header=None,delimiter=r"\s+", names=column_names)
print(housing.head(10))

housing = housing[~(housing['MEDV'] >= 50.0)]

plt.figure(figsize=(20, 10))
sns.heatmap(housing.corr(),  annot=True)

housing = housing.astype(float)
column_best = ['LSTAT', 'PIRATIO', 'TAX', 'RM', 'NOX', 'INDUS']

Xs = housing.loc[:,column_best]
Ys = housing['MEDV']
Xs = Xs / np.max(Xs, axis=0)
Ys = Ys/np.max(Ys, axis=None)

one = np.ones((len(Xs),1))
Xs = np.append(one, Xs, axis=1)
Ys = np.array(Ys).reshape((len(Ys),1))

def traintest_split(Xs, Ys, split):
    indices = np.array(range(len(Xs)))
    train_size = round(split * len(Xs))
    random.shuffle(indices)
    train_indices = indices[0:train_size]
    test_indices = indices[train_size:len(Xs)]
    
    X_train = Xs[train_indices, :]
    X_test = Xs[test_indices, :]
    Y_train = Ys[train_indices, :]
    Y_test = Ys[test_indices, :]
    
    return X_train, Y_train, X_test, Y_test

split = 0.8
X_train, Y_train, X_test, Y_test = traintest_split(Xs, Ys, split)

#normal equation (beta needed)
def normal_equation(X, Y):
    beta = np.dot((np.linalg.inv(np.dot(X.T,X))), np.dot(X.T,Y))

    return beta

def predict(X_test, beta):
    return np.dot(X_test, beta)

beta = normal_equation(X_train, Y_train)
Y_predictions = predict(X_test, beta)

def metrics(Y_predictions, Y_test):

    #calculating mean absolute error
    MAE = np.mean(np.abs(Y_predictions-Y_test))

    #calculating root mean square error
    MSE = np.square(np.subtract(Y_test,Y_predictions)).mean() 
    RMSE = math.sqrt(MSE)

    #calculating r_square
    rss = np.sum(np.square((Y_test - Y_predictions)))
    mean1 = np.mean(Y_test)
    mean2 = np.mean(Y_train)
    sst1 = np.sum(np.square(Y_test-mean1))
    sst2 = np.sum(np.square(Y_test-mean2))
    r_square_test = 1 - (rss/sst1)
    r_square_train = 1 - (rss/sst2)
    
    return MAE, RMSE, r_square_test, r_square_train

mae, rmse, r_square_test, r_square_train = metrics(Y_predictions, Y_test)
print("Mean Absolute Error: ", mae)
print("Root Mean Square Error: ", rmse)
print("R square (training):", r_square_train)
print("R square (testing): ", r_square_test)

Y_converted_predictions = (Y_predictions * 48.8)
Y_converted_test = (Y_test * 48.8)

# actual v predictions
plt.scatter(Y_predictions, Y_test)
plt.title('prediction v actual')
plt.xlabel('actual')
plt.ylabel('prediction')
