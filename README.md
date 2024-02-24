# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given data.

## Program:
```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: MANOJ G
RegisterNumber:  212222240060
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error as mae,mean_squared_error as mse
df=pd.read_csv("student_scores.csv")
df.head()
df.tail()
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
plt.scatter(x_train,y_train,color='purple')
plt.plot(x_train,regressor.predict(x_train),color='yellow')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:
![1-1](https://github.com/Danielmanoj/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/69635071/6a351970-736c-4b05-ac81-72fbf737b51a)


![2](https://github.com/Danielmanoj/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/69635071/be3d94f5-d75a-4dc1-9a6d-111e32214ac7)


![3](https://github.com/Danielmanoj/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/69635071/c79ea166-9ef9-4de9-ad8e-d2e0f9002a88)


![4](https://github.com/Danielmanoj/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/69635071/c9025e52-24da-461f-841e-7debfc0ad989)


![5](https://github.com/Danielmanoj/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/69635071/bffd6107-b906-47ac-8ec8-cfe5d4b44442)



![6](https://github.com/Danielmanoj/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/69635071/aa3b8b77-96d8-41ab-bf46-edbd7498d3f3)



![7-1](https://github.com/Danielmanoj/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/69635071/d27037f7-3ed6-48ef-9ec5-0d941f67cf6e)


![8](https://github.com/Danielmanoj/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/69635071/2aaac4fd-1cd4-4e6d-a0c2-252de9239a09)











## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
