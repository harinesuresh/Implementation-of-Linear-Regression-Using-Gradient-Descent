# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required library and read the dataframe. 
2.Write a function computeCost to generate the cost function. 
3.Perform iterations og gradient steps with learning rate. 
4.Plot the Cost function using Gradient Descent and generate the required graph.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: HARINE S
RegisterNumber: 212224230081
*/
import numpy as np
import matplotlib.pyplot as plt
X = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([2, 4, 5, 4, 5], dtype=float)
m = 0  # slope
b = 0  # intercept
learning_rate = 0.01
epochs = 1000
n = len(X)
for i in range(epochs):
    y_pred = m * X + b
    dm = (-2/n) * np.sum(X * (y - y_pred))
    db = (-2/n) * np.sum(y - y_pred)
    m = m - learning_rate * dm
    b = b - learning_rate * db
print("Slope (m):", m)
print("Intercept (b):", b)
y_pred = m * X + b
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, y_pred, color='red', label='Regression Line')
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()
```

## Output:
<img width="1046" height="866" alt="image" src="https://github.com/user-attachments/assets/1398e2c2-c5ef-4789-bfc1-dfcfb93e0a9d" />

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
