import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

path = 'C:\\Users\\melan\\anaconda3\\PyCharmProject\EdxCourse\\Module3_automobileEDA.csv'

df = pd.read_csv(path)
print(df.head())

# Linear Regression and Multiple Linear Regression
# Simple Linear Regression is a method to help us understand the relationship between two variables:
    The predictor/independent variable (X)
    The response/dependent variable (that we want to predict)(Y)

Linear function:
Y_hat = a + bX, where "a" is the intercept, abd "b" is the slope of the regression line

# Load module from sklearn.linear_model for Linear Regression
from sklearn.linear_model import LinearRegression

# Create the linear regression object:
lm = LinearRegression()
lm

# Using simple linear regression, we will create a linear function with "highway-mpg" as the predictor variable and the "price" as the response variable.

X = df[['highway-mpg']] # use 2 [] for X otherwise we have issues
Y = df['price']

# We can output a prediction:
Yhat = lm.predict(X)
Yhat[0:5] # By using Yhat[0:5] I am slicing the Yhat values to get only the first 5 values.

# value of the intercept (a)
lm.intercept_

# value of the slope (b)
lm.coef_

#Final estimated linear model:
# Yhat = a + bX
Yhat = 38423.31 - 821.73*X
print(Yhat)

# Second method using np.polyfit(x, y, 1)
x = df['highway-mpg'] # In this case only 1 [] for this paramter as we are in 1D polynomial
y = df['price']
lm = np.polyfit(x, y, 1)
print(lm) 

# array results is [a, b] because it follows the function y = a*x + b, where a is the coef and b the intercept 


# LinearRegression object called "lm1" and train the model using "engine-size" as the independent variable and "price" as the dependent variable
from sklearn.linear_model import LinearRegression
lm1 = LinearRegression()
print(lm1)

X = df[['engine-size']]
Y = df['price']
lm1.fit(X,Y)
Yhat = lm1.predict(X)

a = lm1.intercept_
b = lm1.coef_
print("Intercept is", a, "and coef is", b)

Yhat = -7963.34 + 166.86*X
print(Yhat)

# Multiple Linear Regression is very similar to Simple Linear Regression, but this method is used to explain the relationship between 
# one continuous response (dependent) variable: Y variable and two or more predictor (independent) variables: X variables.
# the equation is given by: Yhat = a + b_1X_1 + b_2X_2 + b_3X_3 + b_4X_4

# develop a model using below variables as the predictor variables:
Z = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]

# Fit the linear model using the four above-mentioned variables
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(Z, df['price'])
a = lm.intercept_
b = lm.coef_
print("The intercept a is", a, "the coef b1, b2, b3, b4 are respectively", b) # in this case we have 4 variables

Yhat = lm.predict(Z)
Yhat = -15806.62 + 53.50*df['horsepower'] + 4.71*df['curb-weight'] + 81.53*df['engine-size'] + 36.06*df['highway-mpg']
print(Yhat.to_frame())

# Create and train a Multiple Linear Regression model "lm2" where the response variable is "price", and the predictor variable is "normalized-losses" and "highway-mpg"
Z = df[['normalized-losses', 'highway-mpg']]
Y = df['price']
lm2 = LinearRegression()
lm2 = lm2.fit(Z,Y) # Be careful => lm.fit() !!!! No lm.fit = XXXX
Yhat = lm2.predict(Z)
a = lm2.intercept_
b = lm2.coef_
print("the intercept is", a, "the coef b1 and b2 are respectively", b)
Yhat = 38201.31 + 1.50*df['normalized-losses'] -820.45*df['highway-mpg']

# Model Evaluation Using Visualization
import seaborn as sns
%matplotlib inline

# When it comes to simple linear regression, an excellent way to visualize the fit of our model is by using regression plots.
width = 12
lenght = 10
plt.figure(figsize =(width, lenght))
sns.regplot(x = 'highway-mpg', y= 'price', data= df)
plt.ylim(0,)

# We can see from this plot that price is NEGATIVELY correlated to highway-mpg since the regression slope is negative.

# Use the R-squared (R²) to see the accuracy of the model
from sklearn.metrics import r2_score
r2_score(y, x) # always r2_score (y_value, x_value)

# Let's compare this plot to the regression plot of "peak-rpm":
width = 12
lenght = 10
plt.figure(figsize = (width,lenght))
sns.regplot(x= 'peak-rpm', y= 'price', data=df)
plt.ylim(0,)
plt.show()

# Determine the correlation between "peak-rpm", "highway-mpg" and "price"
df[['peak-rpm', 'highway-mpg', 'price']].corr()

# The residual plot represents the error between the actual value and the predicted value. This is a good way to visualize the variance of the data
width = 12
lenght = 10
plt.figure(figsize= (width, lenght))
sns.residplot(x='highway-mpg', y= 'price', data= df)
plt.show()

# We can see from this residual plot that the residuals are not randomly spread around the x-axis, leading us to believe that maybe a non-linear model is more appropriate for this data.

# Create a Residual plot for Peak-rpm value
width = 10
lenght = 12
plt.figure(figsize= (width, lenght))
sns.residplot(x= 'peak-rpm', y= 'price', data= df)
plt.show()

# Multiple Linear Regression
# One way to look at the fit of the model is by looking at the DISTRIBUTION PLOT. We look at the distribution of the fitted values that result from the model and compare it to the distribution of the actual values.

# Make a prediction:
Z = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]
Y = df['price']
lm = lm.fit(Z,Y)
Y_hat = lm.predict(Z)

width = 12
lenght = 10
plt.figure(figsize= (width, lenght))

# define "ax1" which is the curve of the current values for price
ax1= sns.distplot(df['price'], hist= False, color= "r", label= "Actual Value")

# define the " predicted values" using the sns.distplot function:
sns.distplot(Y_hat, hist = False, color= "b", label= "Fitted Value", ax= ax1)

plt.title('Actual Vs Fitted Values for Price')
plt.xlabel('Price (in dollars)')
plt.ylabel('Proportion of cars')

plt.legend()
plt.show()
plt.close()

# Polynomial Regression and Pipelines
# Polynomial regression is a particular case of the general linear regression model or multiple linear regression models. We get non-linear relationships by squaring or setting higher-order terms of the predictor variables.

# Polynomial model using PlotPolly()
def PlotPolly(model, independent_variable, dependent_variable, Name): # refers to the variable PlotPolly(p, x, y, 'highway-mpg')
    x_new = np.linspace(15, 55, 100) # np.linspace(start, stop, num) # np.linspace(df[feature].min(), df[feature].max(), 100)
    y_new = model(x_new)
    
    plt.plot(independent_variable, dependent_variable, '.', x_new, y_new, '-') # Here dot represents original values and the line represents new values.
    plt.title('Polynomial Fit with Matplotlib for Price ~ Length')
    ax = plt.gca() # gca = get current axes. "Current" here means that it provides a handle to the last active axes.
    ax.set_facecolor(((0.898, 0.898, 0.898)))
    fig= plt.gcf() # means get current figure
    plt.xlabel(Name)
    plt.ylabel('Price of cars')
    
    plt.show()
    plt.close()

# Get the variables:
x= df['highway-mpg']
y= df['price']

# Fit the polynomial using the function np.polyfit() and np.poly1d()
f= np.polyfit(x, y, 3)
p= np.poly1d(f)
print(p)

# Plot the function PlotPolly( [np.poly1d()] , x, y, "x name column")
PlotPolly(p, x, y, 'highway-mpg')

# We can already see from plotting that this polynomial model performs better than the linear model. This is because the generated polynomial function "hits" more of the data points.

# Create 11 order polynomial model with the variables x and y from above
x = df['highway-mpg']
y = df['price']

figure = np.polyfit(x, y, 11)
predict = np.poly1d(figure)
print(predict)

PlotPolly(predict, x, y, 'highway-mpg')
# Let's say the highway mpg is 30, we can predict the price:
highway_mpg = 30
predict(highway_mpg)

# Polynomial regression with more than one dimension 
from sklearn.preprocessing import PolynomialFeatures
pr = PolynomialFeatures(degree= 2)
print(pr)

# Define de variable
Z_pr = pr.fit_transform(Z)
print(Z.shape)
print(Z_pr.shape) # After the transformation, there are 201 samples and 15 features

# Pipeline with StandardScaler
# Data Pipelines simplify the steps of processing the data
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Create the pipeline by creating a list of tuples including the name of the model or estimator and its corresponding constructor.
Input= [('Scale', StandardScaler()), ('Polynomial', PolynomialFeatures(include_bias= False)), ('model', LinearRegression())]

# input the list (Input) as an argument to pipe= Pipeline()
pipe= Pipeline(Input)
print(pipe)

# Transform the data type Z to float .astype(float) to avoid conversion error
Z = Z.astype('float')
print(Z.head())

# Normalize the data, perform a transform and fit the model simultaneously
pipe.fit(Z, y)

# Produce a prediction
Y_pipe = pipe.predict(Z)
print(Y_pipe)[0:5]

# Create a pipeline that standardizes the data, then produce a prediction using a linear regression model using the features Z and target y:
Input_2 = [('Scale', StandardScaler()), ('model', LinearRegression())]
pipe = Pipeline(Input_2)
print(pipe)

pipe.fit(Z, y)

Y_pipe2 = pipe.predict(Z)
print(Y_pipe2)[0:5]

# Measures for In-Sample Evaluation
# R-squared: also known as the coefficient of determination, is a measure to indicate how close the data is to the fitted regression line.
# The closer it is to 1 the more accurate your linear regression model is.
# Mean Squared Error (MSE):measures the average of the squares of errors. That is, the difference between actual value (y) and the estimated value (ŷ).
# The Mean Squared Error tells you how close a regression line is to a set of points (the smaller it is, the better)

# Model 1: Simple Linear Regression with R² and MSE
# define the variables:
from sklearn.linear_model import LinearRegression
lm = LinearRegression()

#highway_mpg_fit
x_model_evaluation = df[['highway-mpg']] # refers at the first line codes for linear regression
y_model_evaluation = df['price']

lm.fit(x_model_evaluation, y_model_evaluation)
print("The R-square is", lm.score(x_model_evaluation, y_model_evaluation))

Yhat = lm.predict(x_model_evaluation)
print('The output of the first 4 predicted value is:', Yhat[0:4])

# import the function mean_squared_error from the module metrics:
from sklearn.metrics import mean_squared_error

# We can compare the predicted results with the actual results:
mse = mean_squared_error(df['price'], Yhat)
print('The mean square error of price and predicted value is:', mse)

# Model 2: Multiple Linear Regression with R² and MSE
# fit the model
lm.fit(Z, df['price'])

# Find the R^2
print('The R-square is', lm.score(Z, df['price']))

# produce prediction:
Y_predict_multifit = lm.predict(Z)

# We compare the predicted results with the actual results:
print('The mean square error of price and predicted value using multifit is:', mean_squared_error(df['price'], Y_predict_multifit))

# Model 3: Polynomial Fit
# Calculate R²
from sklearn.metrics import r2_score
r_squared = r2_score(y, p(x))
print('The R-square value is: ', r_squared)

# calculate MSE
mean_squared_error(df['price'], p(x))

# Prediction and Decision Making
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline

# Create a new input:
new_input = np.arange(1, 100, 1).reshape(-1, 1)

# Fit the model:
lm.fit(X, Y)
lm

# Produce a prediction:
Yhat = lm.predict(new_input)
print(Yhat)[0:5]

# plot the data:
plt.plot(new_input, Yhat)
plt.show()

