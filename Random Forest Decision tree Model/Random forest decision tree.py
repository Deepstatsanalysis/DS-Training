#Data PreProcessing 
#Step1 :
#import basic library

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#connect data
ds=pd.read_csv('Position_Salaries.csv')

#dependent & independent variable
x=ds.iloc[:,1:2].values
y=ds.iloc[:,2].values

         
#Feature scaling 

#import regression library & fit 
from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=500,random_state=0)
regressor.fit(x,y)

#plot graph
#plt.scatter(x,y,color='red')
#plt.plot(x,y,color='blue')
#plt.title('salary v/s level')
#plt.xlabel('level')
#plt.ylabel('salary')
#plt.show()

#plot chart high definition
x_grid=np.arange(min(x),max(x),0.1)
x_grid=x_grid.reshape((len(x_grid),1))



plt.scatter(x,y,color='red')
plt.plot(x_grid,regressor.predict(x_grid),color='blue')
#plt.plot(x,regressor.predict(x),color='blue')
plt.title('Level v/s Salary : Random Forest')
plt.xlabel('Label')
plt.ylabel('Salary')
plt.show()

y_pred=regressor.predict(6.5)
