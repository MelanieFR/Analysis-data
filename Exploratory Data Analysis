import pandas as pd
import numpy as np

# load the data from the DataFrame
filename = 'C:\\Users\\melan\\anaconda3\\PyCharmProject\\EdxCourse\\Module3_automobileEDA.csv'
df = pd.read_csv(filename)
print(df)

# Analyzing Individual Feature Patterns Using Visualization with sns and plt
# When visualizing individual variables, it is important to first understand what type of variable you are dealing with. 
%%capture

import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline # Dont forget %matplotlib inline 

print(df.dtypes)
print(df.info())

# can calculate the correlation between variables of type "int64" or "float64" using the method .corr()
print(df.corr())

# correlation between the following columns: bore, stroke, compression-ratio, and horsepower:
print(df[["bore", "stroke", "compression-ratio", "horsepower"]].corr())

# Continuous Numerical Variables
# Continuous numerical variables are variables that may contain any value within some range. They can be of type "int64" or "float64". 
# A great way to visualize these variables is by using scatterplots with fitted lines.

sns.regplot(x = "engine-size", y = "price", data =df)
plt.ylim(0,)

# As the engine-size goes up, the price goes up: this indicates a positive direct correlation between these two variables.
# confirm the plt results with calculation:
print(df[["engine-size", "price"]].corr())

# Is highway mpg a potential predictor variable of price?
sns.regplot(x = "highway-mpg", y = "price", data =df)
plt.ylim(0,)

print(df[["highway-mpg", "price"]].corr())

# is "peak-rpm" a predictor variable of "price"?
sns.regplot(x = "peak-rpm", y = "price", data =df)
plt.ylim(0,)

print(df[["peak-rpm", "price"]].corr())

# What about the correlation between "stroke" and "price"?
sns.regplot(x = "stroke", y= "price", data= df)
plt.ylim(0,)

print(df[["stroke", "price"]].corr())

# Categorical Variables
# These are variables that describe a 'characteristic' of a data unit, and are selected from a small group of categories. 
# The categorical variables can have the type "object" or "int64". 
# Box plot ideal to visualize categorical variables sns.boxplot(x, y, data)

sns.boxplot(x = "body-style", y= "price", data =df)
plt.ylim(0,)

# the distribution of price between the different body-style categories has a significant overlap, so body-style IS NOT a good predictor of price. 

# Let's examine engine "engine-location" and "price"
sns.boxplot(x= "engine-location", y= "price", data= df)
plt.ylim(0,)

# Here we see that the distribution of price between these two engine-location categories, front and rear, are distinct enough to take engine-location as a potential good predictor of price.

# Let's examine "drive-wheels" and "price"
sns.boxplot(x= "drive-wheels", y= "price", data= df)
plt.ylim(0,)

# Descriptive Statistical Analysis using .describe() method:
print(df.describe())
print(df.describe(include = ['object'])) # to get variables of type "object"

# Convert the series to a dataframe using df['column'].value_counts().to_frame():
df["drive-wheels"].value_counts().to_frame()

# Let's repeat the above steps but save the results to the dataframe "drive_wheels_counts" and rename the column 'drive-wheels' to 'value_counts'.
drive_wheels_counts = df['drive-wheels'].value_counts().to_frame()
drive_wheels_counts.rename(columns= {'drive-wheels':'value_counts'},  inplace = True)
print(drive_wheels_counts)

# Rename the index to 'drive-wheels':
drive_wheels_counts.index.name = 'drive-wheels'
print(drive_wheels_counts)

# Basics of Grouping using.groupby( ) method:
# The "groupby" method groups data by different categories. The data is grouped based on one or several variables, and analysis is performed on the individual groups.

# Step1: Group by the variable using .unique()
print(df['drive-wheels'].unique())

# Select the columns 'drive-wheels', 'body-style' and 'price', then assign it to the variable "df_group_one".
df_group_one = df[['drive-wheels', 'body-style', 'price']]

# Step2: Calculate the avg price of each of the diff categories
df_group_one = df_group_one.groupby(['drive-wheels'], as_index=False).mean()
print(df_group_one)

# Step3: Group by multiple variables
df_gptest = df[['drive-wheels', 'body-style', 'price']]
grouped_test1 = df_gptest.groupby(['drive-wheels', 'body-style'], as_index= False).mean()
print(grouped_test1)

# Pivot table
# This grouped data is much easier to visualize when it is made into a pivot table.
grouped_pivot = grouped_test1.pivot(index = 'drive-wheels', columns= 'body-style')
print(grouped_pivot)

# Use the .fillna(0) method to change NaN:
grouped_pivot.fillna(0)
print(grouped_pivot)

# Use the "groupby" function to find the average "price" of each car based on "body-style"
df_gptest2 = df[['drive-wheels', 'body-style', 'price']]
grouped_test_bodystyle = df_gptest2.groupby(['body-style'], as_index= False).mean()
print(grouped_test_bodystyle)

# Heatmap plots using plt.pcolor( )
import matplotlib.pyplot as plt
%matplotlib inline

#  Let's use a heat map to visualize the relationship between Body Style vs Price.
plt.pcolor(grouped_pivot, cmap= 'RdBu')
plt.colorbar()
plt.show()

# The heatmap plots the target variable (price) proportional to colour with respect to:
        # the variables 'drive-wheel' on the vertical axis
        # and 'body-style' on the horizontal axis.
# This allows us to visualize how the price is related to 'drive-wheel' and 'body-style'.

# Change the default labels
fig, ax= plt.subplots()
im = ax.pcolor(grouped_pivot, cmap= 'RdBu')

# label names:
row_labels = grouped_pivot.columns.levels[1]
col_labels = grouped_pivot.index

# move ticks and labels to the center
ax.set_xticks(np.arange(grouped_pivot.shape[1])+ 0.5, minor=False)
ax.set_yticks(np.arange(grouped_pivot.shape[0])+ 0.5, minor= False)

#insert labels
ax.set_xticklabels(row_labels, minor= False)
ax.set_yticklabels(col_labels, minor= False)

#rotate label if too long
plt.xticks(rotation= 90)

fig.colorbar(im)
plt.show()

# Calculate the Pearson Correlation Coefficient and P-value of variables using stats.pearsonr( )
from scipy import stats

# The Pearson Correlation measures the linear dependence between two variables X and Y.
# 1: Perfect positive linear correlation
# -1: Perfect negative linear correlation.

# The P-value is the probability value that the correlation between two variables is statistically significant.
# P-value < 0.001: Strong certainty in the results

# Calculate the Pearson Correlation Coefficient and P-value of 'wheel-base' and 'price':
pearson_coef, p_value= stats.pearsonr(df['wheel-base'], df['price'])
print("The Pearson Coefficient is:", pearson_coef, " with a P-value of P=", p_value)

# The linear Relationship between Price and Wheel base is not very strong (Pearson coef (~0.59)) while the correlation between price and base wheels is statistically strong (P-value< 0.01)

# relationship between 'wheel-base' and 'price'
sns.regplot(x = 'wheel-base', y= 'price', data= df)
plt.ylim(0,)

# Pearson Correlation Coefficient and P-value:
# Horsepower vs. Price
pearson_coef, p_value= stats.pearsonr(df['horsepower'], df['price'])
print("The Pearson Coefficient is:", pearson_coef, " with a P-value of P=", p_value)

# Length vs. Price
pearson_coef, p_value= stats.pearsonr(df['length'], df['price'])
print("The Pearson Coefficient is:", pearson_coef, " with a P-value of P=", p_value)

# Width vs. Price
pearson_coef, p_value= stats.pearsonr(df['width'], df['price'])
print("The Pearson Coefficient is:", pearson_coef, " with a P-value of P=", p_value)

# Curb-Weight vs. Price
pearson_coef, p_value = stats.pearsonr(df['curb-weight'], df['price'])
print("The Pearson Coefficient is:", pearson_coef, " with a P-value of P=", p_value)

# City-mpg vs. Price
pearson_coef, p_value= stats.pearsonr(df['city-mpg'], df['price'])
print("The Pearson Coefficient is:", pearson_coef, " with a P-value of P=", p_value)

# ANOVA: Analysis of Variance (F-test and P-value) with stats.f_oneway( )
# The Analysis of Variance (ANOVA) is a statistical method used to test whether there are significant differences between the means of two or more groups
grouped_test2 = df_gptest[['drive-wheels', 'price']].groupby(['drive-wheels'])
print(grouped_test2.head())
print(df_gptest)

# We can obtain the values of the method group using the method "get_group".
grouped_test2.get_group('4wd')['price']

# Use the function 'f_oneway' in the module 'stats' to obtain the F-test score and P-value.
f_val, p_val = stats.f_oneway(grouped_test2.get_group('fwd')['price'], grouped_test2.get_group('rwd')['price'], grouped_test2.get_group('4wd')['price'])
print("ANOVA results: F=", f_val, ", P =", p_val)

# This is a great result with a large F-test score showing a strong correlation and a P-value of almost 0 implying almost certain statistical significance.

# fwd and rwd
f_val, p_val = stats.f_oneway(grouped_test2.get_group('fwd')['price'], grouped_test2.get_group('rwd')['price'])
print("ANOVA results: F=", f_val, ", P =", p_val)

# 4wd and rwd
f_val, p_val = stats.f_oneway(grouped_test2.get_group('4wd')['price'], grouped_test2.get_group('rwd')['price'])
print("ANOVA results: F=", f_val, ", P =", p_val)

# 4wd and fwd
f_val, p_val = stats.f_oneway(grouped_test2.get_group('4wd')['price'], grouped_test2.get_group('fwd')['price'])
print("ANOVA results: F =", f_val, ", P =", p_val)






