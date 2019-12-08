# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import the dataset
dataset=pd.read_csv('Salary-Data.csv')
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values

#spliting the dataset into trainig and testing
from sklearn.model_selection import train_test_split
 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

#design the model
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

#prediction 
y_pred=regressor.predict(x_test)

#visualize data
plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('salary vs year of experience')
plt.xlabel('year of experience')
plt.ylabel('salary')
plt.show()

plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('salary vs year of experience')
plt.xlabel('year of experience')
plt.ylabel('salary')
plt.show()