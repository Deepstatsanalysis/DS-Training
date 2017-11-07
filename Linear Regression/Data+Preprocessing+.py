
# coding: utf-8

# In[1]:

#Data Perprocessing 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:

#import csv file
dataset = pd.read_csv('data/data.csv')


# In[3]:

dataset


# In[4]:

x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values


# In[5]:

x


# In[6]:

y


# In[7]:

# X - independent variable & Y is dependent varible 


# In[8]:

# missing values


# In[9]:

from sklearn.preprocessing import Imputer  
imputer=Imputer(missing_values='NaN', strategy='mean', axis=0)         
imputer = imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])


# In[ ]:

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x=LabelEncoder()
labelencoder_y=LabelEncoder()
#Creating error in country encoding so we will use OneHotEncoder
x[:,0]=labelencoder_x.fit_transform(x[:,0])
y=labelencoder_y.fit_transform(y)
#Dummy encoding 
onehotencoder=OneHotEncoder(categorical_features=[0])
x=onehotencoder.fit_transform(x).toarray()

