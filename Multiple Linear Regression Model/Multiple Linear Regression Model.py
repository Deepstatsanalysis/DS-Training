#import library 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import dataset
ds=pd.read_csv('Company_details.csv')

#create matrix for independent & dependent variable 
x=ds.iloc[:,:-1].values
y=ds.iloc[:,4].values   
         
#independent set state is categorical variable to convert into dummy date 
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_x=LabelEncoder() 
x[:,3]= labelencoder_x.fit_transform(x[:,3])
onehotencoder_x=OneHotEncoder(categorical_features=[3]) 
x =onehotencoder_x.fit_transform(x).toarray()



#dummy variable trap need to remove , remove 1 column aways manually but 
x=x[:,1:]

#Split dataset into training set and test set
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)

#Creating Regression Model
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

#prediction on test set
y_pred=regressor.predict(x_test)



#Backward elimination
import statsmodels.formula.api as sm
#if column append then axis=1, otherwise if row add then axis=0
#y=bo+b1x1+b2x2^2+.... , we need to append bo means x=1 that why added 1 
#in first column in 50 lines
x=np.append(arr=np.ones((50,1)).astype(int),values=x,axis=1)

#Now we need to optimize this model so create optimal matrix
x_opt=x[:,[0,1,2,3,4,5]]

regressor_ols=sm.OLS(endog = y, exog = x_opt).fit()

regressor_ols.summary()


#**************
x_opt=x[:,[0,1,3,4,5]]

regressor_ols=sm.OLS(endog = y, exog = x_opt).fit()

regressor_ols.summary()

#************************
x_opt=x[:,[0,3,4,5]]

regressor_ols=sm.OLS(endog = y, exog = x_opt).fit()

regressor_ols.summary()
#***************************
x_opt=x[:,[0,3,5]]

regressor_ols=sm.OLS(endog = y, exog = x_opt).fit()

regressor_ols.summary()
#******************************

x_opt=x[:,[0,3]]

regressor_ols=sm.OLS(endog = y, exog = x_opt).fit()

regressor_ols.summary()







