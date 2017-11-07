
# coding: utf-8

# In[1]:

#Data Perprocessing 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:

#import csv file
dataset = pd.read_csv('data/Salary_Data.csv')


# In[3]:

dataset


# In[4]:

x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values


# In[5]:

x


# In[6]:

y


# In[8]:

#split dataset into training set & Test Set
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)


# In[9]:

#Creating Simple Linear Regressor Model for Training set 
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)


# In[10]:

#Predict salary for test set data point
y_pred=regressor.predict(x_test)


# In[11]:

#Visualize chart for training set
plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('salary v/s Exper (training set)')
plt.xlabel('Year of exp')
plt.ylabel('Salary')
plt.show()


# In[14]:

#Visualize chart for Test set
plt.scatter(x_test,y_test,color='Red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('salary v/s exp (Test set)')
plt.xlabel('year of exper')
plt.ylabel('Salary')
plt.show()

