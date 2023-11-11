# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Prepare your data -Collect and clean data on employee salaries and features -Split data into training and testing sets

2.Define your model -Use a Decision Tree Regressor to recursively partition data based on input features -Determine maximum depth of tree and other hyperparameters

3.Train your model -Fit model to training data -Calculate mean salary value for each subset

4.Evaluate your model -Use model to make predictions on testing data -Calculate metrics such as MAE and MSE to evaluate performance

5.Tune hyperparameters -Experiment with different hyperparameters to improve performance

6.Deploy your model Use model to make predictions on new data in real-world application.

## Program:
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

Developed by:MADHUMITHA M

Register Number:212222220020  

import pandas as pd

df=pd.read_csv("Salary.csv")


df.head()

df.info()

df.isnull().sum()

from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

df['Position']=le.fit_transform(df['Position'])

df.head()

x=df[['Position','Level']]

y=df['Salary']


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)


from sklearn.tree import DecisionTreeRegressor

dt=DecisionTreeRegressor()

dt.fit(x_train,y_train)

y_pred=dt.predict(x_test)


from sklearn import metrics

mse=metrics.mean_squared_error(y_test,y_pred)

mse

r2=metrics.r2_score(y_test,y_pred)

r2

dt.predict([[5,6]])

## Output:

Initial dataset:

![image](https://github.com/Madhumithamahendran/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119394403/0431361c-edd2-43be-b055-ba4836b75c81)

Data info:

![image](https://github.com/Madhumithamahendran/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119394403/62746077-ed08-430d-ba21-04f9746c8bce)

Optimization of null values:

![image](https://github.com/Madhumithamahendran/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119394403/92665bb3-6204-4ca0-a246-93ec99f55593)

Converting string values to numerical values using lab encoder:

![image](https://github.com/Madhumithamahendran/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119394403/294eb15a-e3dc-4001-8723-ecc4099484a3)

Mean squared error:

![image](https://github.com/Madhumithamahendran/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119394403/363b909b-983a-4575-a7f1-35e92efc5c56)

R2(variance):

![image](https://github.com/Madhumithamahendran/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119394403/008158b6-0683-4b13-bf91-c7fb117339c4)

Prediction:

![image](https://github.com/Madhumithamahendran/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119394403/049acdc5-9187-4e39-8d6b-efad7a7b9b6d)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
