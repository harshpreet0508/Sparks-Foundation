# DATA SCIENCE AND BUSINESS ANALYTICS at SPARKS FOUNDATION

# Author: Harshpreet Kaur

#----------------------------------------------------------------------
# TASK 1: Prediction using Supervised ML

# Predict the percentage of an student based on the no. of study hours.

# A simple Linear Regression task that involves two variables.
#-----------------------------------------------------------------------

#--Importing the Header Files-- 

import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
  

#--Get the data--

dataset = pd.read_csv("C:\\Users\\harsh\\OneDrive\\Desktop\\Sparks Foundation\\Datasets\\scores.csv")
print("Data imported successfully")


#--Data Preprocessing--

# printing head( first few rows) of the dataset 
dataset.head()

# the datatype of the columns
dataset.dtypes

# the total no of rows and columns
dataset.shape

# basic statisical details 
dataset.describe()

# checking for any null values
dataset.isnull().sum()


#--Data Visualization--

# Plotting the distribution of scores
dataset.plot(x='Hours', y='Scores', style='o')
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()

# The above graph clearly shows that there is a linear relation between the number of hours studied and percentage score obtained.


#--Preparing the data-- 

# The next step is to split the data into X (feature matrix or input) and y (vector of predictions or output) using the iloc (integer location) function.

X = dataset.iloc[:, :-1].values  
y = dataset.iloc[:, 1].values 

X.reshape(-1,1)
y.reshape(-1,1)


#--Splitting dataset into training and test set--

# We will be doing this by using Scikit-Learn's built-in train_test_split() method. 

from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y) 


#--Select and Train an ML Algorithm--
# For this task its the linear regression Algorithm 

from sklearn.linear_model import LinearRegression  
lin_reg = LinearRegression() 

# to tell the also which data to work on we use the fit function
lin_reg.fit(X_train, y_train) 

# Visualising the Training dataset 
plt.scatter(X_train,y_train)
plt.title('Training set')  
plt.plot(X_train,lin_reg.predict(X_train))
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score') 
plt.show()

# Accuracy of training set
lin_reg.score(X_train, y_train) 

# Now, finding the equation y = mx + c

# Plotting the regression line
line = lin_reg.coef_*X + lin_reg.intercept_
plt.scatter(X, y)
plt.title('Regression Line') 
plt.plot(X, line)
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score') 
plt.show()


#--Making Predictions--
# Now that we have trained our algorithm, it's time to make some predictions.

print(X_test) # Testing data - In Hours
y_pred = lin_reg.predict(X_test) # Predicting the scores

# Comparing Actual vs Predicted
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df 


# Now, what will be the score if a student studies 9.25 hours per day ? 

hours = float(input('Enter number of hours a student in studying in a day '))

own_pred = lin_reg.predict([[hours]])
print("Predicted Score = {}".format((own_pred)[0]))


#--Evaluation of the Model--
# As the final step we find the accuracy of the test set, visualize it and find the mean squared error.

# Accuracy of test set
lin_reg.score(X_test, y_test) 

from sklearn import metrics  
print('Mean Absolute Error:',metrics.mean_absolute_error(y_test, y_pred)) 

# Visualising the test dataset 
plt.scatter(X_test,y_test)
plt.title('Test set')  
plt.plot(X_train,lin_reg.predict(X_train))
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score') 
plt.show()

#---------------------------------

# Therefore, Task 1 is complete.

# Thank you !!

#---------------------------------





