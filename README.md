# DiabetesDatasets
#LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
data_train=np.loadtxt("Train_data.csv",delimiter=',')
x_train=data_train[:,:-1]
y_train=data_train[:,-1]
data_test=np.loadtxt("Test_data.csv",delimiter=',')
x_test=data_test
alg1=LinearRegression()
alg1.fit(x_train,y_train)
y_pred=alg1.predict(x_test)
y_pred=y_pred.round(5)
np.savetxt("Prediction_data_here.csv",y_pred)
