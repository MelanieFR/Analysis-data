# Load the data and libraries

import pandas as pd
import numpy as np

filename = 'C:\\Users\\melan\\anaconda3\\PyCharmProject\\EdxCourse\\module5_auto.csv'
df = pd.read_csv(filename)

df.to_csv('module5_auto_reimport.csv')

# First lets use only numeric data using the function ._get_numeric_data()
df = df._get_numeric_data()
print(df.head())

%%capture
from ipywidgets import interact, interactive, fixed, interact_manual

# Functions for Plotting
def DistributionPlot(RedFunction, BlueFunction, RedName, BlueName, Title):
    width = 12
    lenght = 10
    plt.figure(figsize=(width, lenght))
    
    ax1 = sns.distplot(RedFunction, hist= False, color= "r", label= RedName)
    ax2 = sns.distplot(BlueFunction, hist = False, color= "b", label= BlueName, ax=ax1)
    
    plt.title(Title)
    plt.xlabel('Price (in dollars)')
    plt.ylabel('Proportion of cars')
    
    plt.legend()
    plt.show()
    plt.close()

def PollyPlot(xtrain, xtest, y_train, y_test, lr, poly_transform):
    plt.figure(figsize=(12,10))
    
    # xtrain, y_train: training data
    # xtest, y_test: testing data
    # lr: Linear Regression object
    # poly_transform: Polynomial Transformation object
    
    xmax= max([xtrain.values.max(), xtest.values.max()])
    xmin= min([xtrain.values.min(), xtest.values.min()])
    x=np.arange(xmin, xmax, 0.1)
    
    plt.plot(xtrain, y_train, 'ro', label= 'Training Data')
    plt.plot(xtest, y_test, 'go', label= 'Testing Data')
    plt.plot(x, lr.predict(poly_transform.fit_transform(x.reshape(-1, 1))), label= 'Predicted Function')
    plt.ylim(-10000, 60000)
    plt.ylabel('Price')
    plt.legend()
    
    
# Part 1: Training and Testing
# An important step in testing your model is to split your data into training and testing data. We will place the target data "price" in a separate dataframe "y_data":
y_data = df['price']

# Drop price data in dataframe x_data:
x_data = df.drop('price', axis= 1)

# Now, we randomly split our data into training and testing data using the function train_test_split.
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size= 0.10, random_state=1)

print("number of test samples is", x_test.shape[0])
print("number of training samples is", x_train.shape[0])

# Use the function "train_test_split" to split up the dataset such that 40% of the data samples will be utilized for testing. Set the parameter "random_state" equal to zero. 
x_train1, x_test1, y_train1, y_test1 = train_test_split(x_data, y_data, test_size=0.4, random_state=0)
print("number of test samples is", x_test1.shape[0]) # shape = 0, it refers to "x_data" 
print("number of training samples is:", x_train1.shape[0])

# Import Linear Regression module
from sklearn.linear_model import LinearRegression

# Create a Linear Regression object
lre = LinearRegression()

# Fit the model using 'horsepower'
print(lre.fit(x_train[['horsepower']], y_train))

# calculate R² on the test data
print(lre.score(x_test[['horsepower']], y_test))

# What is the R² on the train data?
print(lre.score(x_train[['horsepower']], y_train))

# Find the R^2 on the test data using 40% of the dataset for testing
lre1 = LinearRegression()
print(lre1.fit(x_train1[['horsepower']], y_train1))

# Calculate R² of the Test modelonly:
lre1.score(x_test1[['horsepower']], y_test1)
lre1.score(x_train1[['horsepower']], y_train1)

print("the R² of the test model is:", lre1.score(x_test1[['horsepower']], y_test1))
print("the R² of the train model is: ", lre1.score(x_train1[['horsepower']], y_train1))

# Cross-validation
# In this method, the dataset is split into K equal groups. Each group is referred to as a fold.
from sklearn.model_selection import cross_val_score

# We input the object, the model, the feature ("horsepower"), and the target data (y_data). The parameter 'cv' determines the number of folds. In this case, it is 4.
R_cross = cross_val_score(lre, x_data[['horsepower']], y_data, cv= 4)

# The default scoring is R^2. Each element in the array has the average R^2 value for the fold:
print(R_cross)

# Calculate the average and standard deviation of our estimate
print("the mean of the folds are:", R_cross.mean(), "and the standard deviation is:", R_cross.std())

# We can use negative squared error as a score by setting the parameter 'scoring' metric to 'neg_mean_squared_error'.
-1 * cross_val_score(lre, x_data[['horsepower']], y_data, cv= 4, scoring ='neg_mean_squared_error')

# Calculate the average R^2 using two folds, then find the average R^2 for the second fold utilizing the "horsepower" feature:
R_cross1 = cross_val_score(lre, x_data[['horsepower']], y_data, cv= 2)
print(R_cross1)
print(R_cross1.mean())

# Function cross_val_predict()
from sklearn.model_selection import cross_val_predict
yhat = cross_val_predict(lre, x_data[['horsepower']], y_data, cv= 4)
print(yhat)[0:5]

# Part 2: Overfitting, Underfitting and Model Selection
# It turns out that the test data, sometimes referred to as the "out of sample data", is a much better measure of how well your model performs in the real world. 
# One reason for this is overfitting.

# Create Multiple Linear Regression objects and train the model
lr = LinearRegression()
print(lr.fit(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_train))

# Prediction using training data:
Yhat_train = lr.predict(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
print(Yhat_train)[0:5]

# Prediction using test data:
Yhat_test = lr.predict(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
print(Yhat_test[0:5])

# Perform some model evaluation using our training and testing data separately.
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline

Title = 'Distribution  Plot of  Predicted Value Using Training Data vs Training Data Distribution'
DistributionPlot(y_train, Yhat_train, "Actual Values (Train)", "Predicted Values (Train)", Title)
# So far, the model seems to be doing well in learning from the training dataset. 

Title = 'Distribution Plot of Predicted Value using Test Data vs Data Distribution of Test Data'
DistributionPlot(y_test, Yhat_test, "Actual Values (Test)","Predicted Values (Test)",Title)
# Comparing Figure 1 and Figure 2, it is evident that the distribution of the test data in Figure 1 is much better at fitting the data.
# This difference in Figure 2 is apparent in the range of 5000 to 15,000. 
# Let's see if polynomial regression also exhibits a drop in the prediction accuracy when analysing the test dataset.

# import the PolynomialFeatures module preprocessing
from sklearn.preprocessing import PolynomialFeatures

# Overfitting occurs when the model fits the noise, but not the underlying process. 
# Let's use 55 percent of the data for training and the rest for testing:
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.45, random_state= 0) 

# We will perform a degree 5 polynomial transformation on the feature 'horsepower':
pr = PolynomialFeatures(degree =5)

x_train_pr = pr.fit_transform(x_train[['horsepower']])
x_test_pr = pr.fit_transform(x_test[['horsepower']])
print(pr)

# Now, let's create a Linear Regression model "poly" and train it.
poly = LinearRegression()
poly.fit(x_train_pr, y_train)

# We can see the output of our model using the method "predict." We assign the values to "yhat":
Yhat_pr = poly.predict(x_test_pr)
print(Yhat_pr[0:5])

# Let's take the first five predicted values and compare it to the actual targets.
print("Predicted values:", Yhat_pr[0:4])
print("Actual values:", y_test[0:4].values)

# Use the function  "PollyPlot" to display the training data, testing data, and the predicted function
PollyPlot(x_train[['horsepower']], x_test[['horsepower']], y_train, y_test, poly, pr)

# R² of the training data:
print(poly.score(x_train_pr, y_train))

# R^2 of the test data:
print(poly.score(x_test_pr, y_test))

# We see the R^2 for the training data is 0.5567 while the R^2 on the test data was -29.87. 
# The lower the R^2, the worse the model. A negative R^2 is a sign of overfitting.

# Let's see how the R^2 changes on the test data for different order polynomials and then plot the results:
# Step 1: Create an empty list to store the values
Rsqu_test = []

# Step2 :create the list containing different Polynomial orders:
order = [1, 2, 3, 4]

# We then iterate through the list using the loop
for n in order:
    pr = PolynomialFeatures(degree = n) # We create a Polynomial feature object with the order of the Polynomial as a parameter
    x_train_pr= pr.fit_transform(x_train[['horsepower']]) # We transform the training & test data into a polynomial using the fit transform method
    x_test_pr= pr.fit_transform(x_test[['horsepower']])
    
    lr.fit(x_train_pr, y_train) # We fit the regression model using the transform data
    
    Rsqu_test.append(lr.score(x_test_pr, y_test)) # We can calculate the R-squared using the test data and store it in the array
    
plt.plot(order, Rsqu_test)
plt.xlabel('order')
plt.ylabel('R^2')
plt.title('R^2 Using Testing Data')
plt.text(3.1, 0.74, 'Maximum R^2') # define the coordinate (x, y) where to write the text 'Maximum R^

def f(order, test_data):
    x_train, x_test, y_train, y_test= train_test_split(x_data, y_data, test_size = test_data, random_state=0)
    pr= PolynomialFeatures(degree = order)
    x_train_pr = pr.fit_transform(x_train[['horsepower']])
    x_test_pr= pr.fit_transform(x_test[['horsepower']])
    poly= LinearRegression()
    poly.fit(x_train_pr, y_train)
    PollyPlot(x_train[['horsepower']], x_test[['horsepower']], y_train, y_test, poly, pr)

# The following interface allows you to experiment with different polynomial orders and different amounts of data:
interact(f, order =(0, 6,1), test_data=(0.05, 0.95, 0.05))

# Create a "PolynomialFeatures" object "pr1" of degree two:
pr1 = PolynomialFeatures(degree=2)
print(pr1)

# Transform the training and testing samples for the features 'horsepower', 'curb-weight', 'engine-size' and 'highway-mpg'. 
x_train_pr1 = pr1.fit_transform(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
x_test_pr1 = pr1.fit_transform(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])

# How many dimensions does the new feature have?
print(_train_pr1.shape  # there are now 15 features)

# Create a linear regression model "poly1". Train the object using the method "fit" using the polynomial features.
poly1 = LinearRegression()
poly1.fit(x_train_pr1, y_train)

# Use the method "predict" to predict an output on the polynomial features, 
# then use the function "DistributionPlot" to display the distribution of the predicted test output vs. the actual test data.

Yhat_train1 = poly1.predict(x_train_pr1)
Yhat_test1 = poly1.predict(x_test_pr1)

Title='Distribution  Plot of  Predicted Value Using Test Data vs Data Distribution of Test Data'
DistributionPlot(y_test1, Yhat_test1, 'Actual Test values', 'Predicted test values', Title)

# Part 3: Ridge Regression
# Ridge regression is a regression that is employed in a Multiple regression model when Multicollinearity occurs.
# Multicollinearity is when there is a strong relationship among the independent variables.

# Let's perform a degree two polynomial transformation on our data:
pr= PolynomialFeatures(degree= 2)
x_train_pr= pr.fit_transform(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg','normalized-losses','symboling']])
x_test_pr= pr.fit_transform(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg','normalized-losses','symboling']])

# Import the module:
from sklearn.linear_model import Ridge

# Let's create a Ridge regression object, setting the regularization parameter (alpha) to 1:
RidgeModel = Ridge(alpha = 1)

# Like regular regression, you can fit the model using the method fit:
print(RidgeModel.fit(x_train_pr, y_train))

# Similarly, you can obtain a prediction:
Yhat = RidgeModel.predict(x_test_pr)

# Let's compare the first five predicted samples to our test set:
print('Predicted values:', Yhat[0:4])
print('Test set:', y_test[0:4].values) # Use .values to have the value, otherwise it gives the features only

# We select the value of alpha that minimizes the test error. 
# To do so, we can use a for loop. We have also created a progress bar to see how many iterations we have completed so far.

from tqdm import tqdm

Rsqu_test = []
Rsqu_train = []
dummy1 = []
Alpha = 10* np.array(range(0,1000))
pbar= tqdm(Alpha)

for alpha in pbar:
    RidgeModel = Ridge(alpha= alpha)
    RidgeModel.fit(x_train_pr, y_train)
    test_score, train_score = RidgeModel.score(x_test_pr, y_test), RidgeModel.score(x_train_pr, y_train)
    
    pbar.set_postfix({"Test Score": test_score, "Train Score": train_score})
    
    Rsqu_test.append(test_score)
    Rsqu_train.append(train_score)


# We can plot out the value of R^2 for different alphas:

plt.figure(figsize=(12,5))

plt.plot(Alpha, Rsqu_test, label='validation data')
plt.plot(Alpha, Rsqu_train, 'r', label='training Data')
plt.xlabel('Alpha')
plt.ylabel('R^2')
plt.legend()
plt.show()

# Perform Ridge regression. 
# Calculate the R^2 using the polynomial features, use the training data to train the model and use the test data to test the model. 
# The parameter alpha should be set to 10.

from sklearn.linear_model import Ridge
RidgeModel_1 = Ridge(alpha = 10)
RidgeModel_1.fit(x_train_pr, y_train)

Yhat_1 = RidgeModel_1.predict(x_test_pr)
print(Yhat_1[0:5])

print(y_train[0:5].values)
print("The R^2 of the train data is", RidgeModel_1.score(x_train_pr, y_train))
print("The R^2 of the test data is", RidgeModel_1.score(x_test_pr, y_test))

# Part 4: Grid Search
# In the last section, the term alpha in Ridge regression is called a hyperparameter. Scikit-learn has a means of automatically iterating over these hyperparameters using cross-validation called Grid Search.

from sklearn.model_selection import GridSearchCV

parameters1 = [{'alpha': [0.001, 0.1, 1, 10, 100, 1000, 10000, 100000, 100000]}]
print(parameters1)

# Create a Ridge regression object:
RR = Ridge()
print(RR)

Grid1 = GridSearchCV(RR, parameters1, cv=4)

# Fit the model:
Grid1.fit(x_data[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_data)

BestRR= Grid1.best_estimator_
print(BestRR)

print(BestRR.score(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_test))

scores= Grid1.cv_results_
print(scores['mean_test_score'])
