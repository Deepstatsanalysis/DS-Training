import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Read data
ds = pd.read_csv('position_Salaries.csv')
x=ds.iloc[:,1:2].values
y=ds.iloc[:,2].values
         
#Feature scaling is not require in Linear regression model library

#No need to creat test data because data set is small or we can split sampling but here have to verify 
#employee whose designation is 6.5 and he is telling 160000$ ture or false.

#Step 1.1 review Linear Regression
from sklearn.linear_model import LinearRegression
Lgr_model=LinearRegression()
Lgr_model.fit(x,y)

#Step 1.2 Review Linear Regression model
plt.scatter(x,y,color='Red')
plt.plot(x,Lgr_model.predict(x),color='blue')
plt.title('position v/s salaries')
plt.xlabel('position')
plt.ylabel('salaries')
plt.show()
#*******verfiy **************
Lgr_model.predict(6.5)
#Step 2.1 Review in Polynomial Linear Regression
from sklearn.preprocessing import PolynomialFeatures
Lgr_model2=LinearRegression()

p_reg=PolynomialFeatures(degree=4)
x_poly=p_reg.fit_transform(x)
p_reg.fit(x_poly,y)
Lgr_model2.fit(x_poly,y)

#Step2.2 Apply Linear ==============
plt.scatter(x,y,color='Red')
plt.plot(x,Lgr_model2.predict(p_reg.fit_transform(x)),color='blue')
plt.title('position v/s salaries -Polynomial')
plt.xlabel('position')
plt.ylabel('salaries')
plt.show()
#***********************************
Lgr_model2.predict(p_reg.fit_transform(6.5))

