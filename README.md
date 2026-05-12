# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load Data & Visualization: Read the dataset (ex1.txt) containing city population vs. profit data and visualize it using a scatter plot.

2.Prepare Inputs: Add an intercept term to the feature matrix X and reshape the profit values y for linear regression.

3.Initialize Parameters: Set initial theta values (model parameters) to zeros.

4.Compute Cost: Define computeCost() to calculate the cost (error) of predictions using the current theta.

5.Gradient Descent: Implement gradientDescent() to iteratively update theta by minimizing the cost function using the gradient of the error.

6.Model Training: Train the model with learning rate α = 0.01 and 1500 iterations to find optimal theta.

7.Visualization: Plot the cost function convergence and overlay the regression line on the data.

8.Prediction: Define a predict() function to estimate profit for given populations using the trained model.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: HARINE S
RegisterNumber:  212224230081
*/
```
```
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
data=pd.read_csv("/content/ex1.txt",header = None)

plt.scatter(data[0],data[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City(10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")

def computeCost(X,y,theta):
  """
  Take in a numpy array X,y,theta and generate the cost function of using the in a linear regression model
  """
  m=len(y) # length of the training data
  h=X.dot(theta) #hypothesis
  square_err=(h-y)**2

  return 1/(2*m) * np.sum(square_err) #returning J

data_n=data.values
m=data_n[:,0].size
X=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))
computeCost(X,y,theta) #Call the function

from matplotlib.container import ErrorbarContainer
from IPython.core.interactiveshell import error
def gradientDescent(X,y,theta,alpha,num_iters):
    """
    Take the numpy array X,y,theta and update theta by taking the num_tiers gradient with learning rate of alpha

    return theta and the list of the cost of theta during each iteration
    """

    m=len(y)
    J_history=[]

    for i in range(num_iters):
      predictions=X.dot(theta)
      error=np.dot(X.transpose(),(predictions -y))
      descent=alpha *1/m*error
      theta-=descent
      J_history.append(computeCost(X,y,theta))

    return theta,J_history

theta,J_history = gradientDescent(X,y,theta,0.01,1500)
print("h(x)="+str(round(theta[0,0],2))+"+"+str(round(theta[1,0],2))+"x1")

#Testing the implementation
plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")


plt.scatter(data[0],data[1])
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0] for y in x_value]
plt.plot(x_value,y_value,color="r")
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City (10,000s)")
plt.ylabel("Profit($10,000")
plt.title("Profit Prediction"

def predict(x,theta):
  """
  Tkes in numpy array of x and theta and return the predicted value of y base
  """

  predictions=np.dot(theta.transpose(),x)

  return predictions[0]

predict1=predict(np.array([1,3.5]),theta)*10000
print("For population =35,000, we predict a profit of $"+str(round(predict1,0)))

predict2=predict(np.array([1,7]),theta)*10000
print("For population = 70,000, we predict a profit of $"+str(round(predict2,0)))
```

## Output:

Profit Prediction Graph :



<img width="574" height="455" alt="image" src="https://github.com/user-attachments/assets/7d99288d-4200-46f7-9d72-ae8fcecca1f5" />




<img width="570" height="455" alt="image" src="https://github.com/user-attachments/assets/54052aed-522b-4763-823b-ac6303e88832" />




Compute Cost Value :



<img width="242" height="33" alt="image" src="https://github.com/user-attachments/assets/4f00c959-6ae6-4086-afdf-ebe7ed848601" />



h(x) Value :



<img width="235" height="37" alt="image" src="https://github.com/user-attachments/assets/0b6e6541-8e5f-476a-ab95-9443f237d52b" />




Cost function using Gradient Descent Graph :



<img width="719" height="502" alt="image" src="https://github.com/user-attachments/assets/02bdf545-fbcf-426c-9741-10d51a09737d" />




Profit for the Population 35,000 :



<img width="486" height="41" alt="image" src="https://github.com/user-attachments/assets/17983ece-7431-4865-ba2e-7b6c10d4dcf9" />




Profit for the Population 70,000 :



<img width="524" height="37" alt="image" src="https://github.com/user-attachments/assets/2b33f67f-4ef3-441a-80fb-28fd7739c17e" />




## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
