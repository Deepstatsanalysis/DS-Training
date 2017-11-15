
#import library 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset
ds=pd.read_csv('Position_Salaries.csv') 

#independent (X) and dependent variable (Y)
x=ds.iloc[:,1:2].values
y=ds.iloc[:,2].values 

         
#import decision tree algorithm
from sklearn.tree import DecisionTreeRegressor
regressor=DecisionTreeRegressor(random_state=0)

#Fit regressor
regressor.fit(x,y)

#plot graph
plt.scatter(x,y,color='red')
plt.plot(x,y,color='blue')
plt.title('salary v/s level')
plt.xlabel('level')
plt.ylabel('salary')
plt.show()
