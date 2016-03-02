#!/usr/bin/env python

#author:zhanghai
#date: 2015-2-1
#filename: svr-predict1.py

import numpy as np
import math
#from sklearn.linear_model.ftrl_classifier import FtrlClassifier
#from matplotlib import pylab as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor
from sklearn.ensemble.forest import RandomForestRegressor,ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
import time

TRAIN_SIZE = 960
TEST_SIZE = 480

def get_data(file_name):
    Y_parameters = pd.read_csv(file_name).fillna(-1).drop("day",axis=1).values.ravel()
    X_parameters = [ [x] for x in xrange(len(Y_parameters))]
    return X_parameters, Y_parameters

def svr_main(X, Y):
    X_train = X[:TRAIN_SIZE]
    Y_train = Y[:TRAIN_SIZE]
    X_test = X[TRAIN_SIZE:]
    Y_test = Y[TRAIN_SIZE:]

    #clf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    #clf.fit(X_train,Y_train)
    #y_pred = clf.predict(X_test)
    #plt.plot(X_test, y_pred, linestyle='-', color='red') 

    clf = GradientBoostingRegressor(n_estimators=100,max_depth=1)
    #clf = DecisionTreeRegressor(max_depth=25)
    #clf = ExtraTreesRegressor(n_estimators=2000,max_depth=14)
    #clf = xgb.XGBRegressor(n_estimators=2000,max_depth=25)
    #clf = RandomForestRegressor(n_estimators=1000,max_depth=26,n_jobs=7)
    predict_list = []
    for i in xrange(TEST_SIZE):
        X = [ [x] for x in xrange(i, TRAIN_SIZE+i)]
        clf.fit(X, Y[i:TRAIN_SIZE+i])
        y_pred = clf.predict([TRAIN_SIZE+1+i])
        predict_list.append(y_pred)

    print "mean_squared_error:%s"%mean_squared_error(Y_test, predict_list)
    print "sqrt of mean_squared_error:%s"%np.sqrt(mean_squared_error(Y_test, predict_list))
    origin_data = Y_test
    print "origin data:%s"%origin_data
    plt.plot([ x for x in xrange(TRAIN_SIZE+1, TRAIN_SIZE+TEST_SIZE+1)], predict_list, linestyle='-', color='red', label='prediction model')  
    plt.plot(X_test, Y_test, linestyle='-', color='blue', label='actual model') 
    plt.legend(loc=1, prop={'size': 12})
    plt.show()


if __name__ == "__main__":
    start = time.clock()
    X, Y = get_data("data1.csv")
    svr_main(X,Y)
    end = time.clock()
    run_time = end - start
    
    print "The program run time:%s"%run_time
    
