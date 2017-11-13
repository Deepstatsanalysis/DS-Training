#Import Library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Read dataset
ds=pd.read_csv('Position_Salaries.csv')

#Split X & Y Variables
X=ds.iloc[:,1:2].values
Y=ds.iloc[:,2].values
         
#SVR library is not supporting feature scaling so manually manange 

from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
sc_y=StandardScaler()
X=sc_x.fit_transform(X)
Y=sc_y.fit_transform(Y)

#Import SVR library
from sklearn.svm import SVR
regressor=SVR(kernel='rbf')
regressor.fit(X,Y)


#create model
plt.scatter(X,Y,color='red')
plt.plot(X,regressor.predict(X),color='blue')
plt.title('Position v/s Salaries (SVR)')
plt.xlabel('Position')
plt.ylabel('Salaries')
plt.show()

#Predict salary of employee whose experience is 6.5 years

y_pred = sc_y.inverse_transform(regressor.predict(sc_x.fit_transform(np.array([[6.5]]))))


