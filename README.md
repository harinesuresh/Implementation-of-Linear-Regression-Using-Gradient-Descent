# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import NumPy, Pandas, and StandardScaler libraries for numerical operations, dataset handling, and feature scaling.

2.Define a linear regression function that adds a bias column, initializes parameters, and updates them using gradient descent.

3.Read the 50_Startups.csv dataset and display the initial records.

4.Extract the independent variables excluding the categorical column and convert them into floating-point values.

5.Extract the dependent variable and reshape it into a column vector.

6.Apply standard scaling to both the independent and dependent variables.

7.Pass the scaled independent and dependent variables to the gradient descent function to compute optimal parameters.

8.Provide new input data and apply the same scaling technique to it.

9.Compute the predicted output using the learned parameters.

10.Convert the predicted value back to its original scale using inverse transformation.

11.Display the final predicted value.


## Program:
```


Developed by: HARINE S
RegisterNumber:  212224230081


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate = 0.1, num_iters = 1000):
    X = np.c_[np.ones(len(X1)),X1]
    theta = np.zeros(X.shape[1]).reshape(-1,1)
    
    for _ in range(num_iters):
        predictions = (X).dot(theta).reshape(-1,1)
        errors=(predictions - y ).reshape(-1,1)
        theta -= learning_rate*(1/len(X1))*X.T.dot(errors)
        return theta
data=pd.read_csv("50_Startups.csv")
print(data.head())
print("\n")
X=(data.iloc[1:,:-2].values)
X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X)
print("\n")
print(X1_Scaled)
print("\n")
theta= linear_regression(X1_Scaled,Y1_Scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(prediction)
print(f"Predicted value: {pre}")



```

## Output:
<img width="653" height="155" alt="Screenshot 2026-01-30 105603" src="https://github.com/user-attachments/assets/8263a6d9-be33-472c-aac8-9e691024b0e4" />

<img width="536" height="172" alt="image" src="https://github.com/user-attachments/assets/f6206295-e70d-4408-b4cb-cfe363b23cf0" />




## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
