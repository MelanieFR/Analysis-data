import pandas as pd
import numpy as np

# the path under Windows is always written \\
path = 'C:\\Users\\melan\\anaconda3\\PyCharmProject\\EdxCourse\\Module1_auto.csv'

df = pd.read_csv(path, header = None)

# show the first 5 rows using dataframe.head() method
print(df.head())

# add headers to the dataframe
headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]

print("headerss\n", headers)

# Then, we use df.columns = headers to replace the headers with the list we created.
df.columns = headers
print(df.head())

# replace '?' by NaN
df_Nan = df.replace("?", np.nan)

# Drop missing values along the column "price"
df = df_Nan.dropna(subset = ["price"], axis = 0)
prin(df.head(20))

# Find the name of the columns of the dataframe:
print(df.columns)

# save dataframe under different file types
df.to_csv("C:\\Users\\melan\\anaconda3\\PyCharmProject\\EdxCourse\\Module1_automobile.csv")
df.to_excel("C:\\Users\\melan\\anaconda3\\PyCharmProject\\EdxCourse\\Module1_automobile.xlsx")

# To identify the data type of each column
print(df.dtypes)

# Describe method to get a statistical summary of each column e.g. count, column mean value, column standard deviation....
print(df.describe())

# To have all columns included with Nan values:
print(df.describe(include = "all"))

# Describe details of selected column:
df[['length', 'compression-ratio']].describe()

# .info() method provides a concise summary of your DataFrame
print(df.info())



# Data Wrangling
import pandas as pd
import numpy as np
import matplotlib.pylab as plt

filename = 'C:\\Users\\melan\\anaconda3\\PyCharmProject\\EdxCourse\\Module2 - imports-85.data'
df = pd.read_csv(filename, header = None)
headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]
           
df.columns = headers
print(df.head())

# Convert "?" to NaN with df.replace(A, B, inplace = True)
df.replace("?", np.nan, inplace = True)
print(df.head())

# .isnull() method to identify the missing values on the df 
missing_data = df.isnull()
print(missing_data.head())

# Count missing values in each column with the function ".value_counts()"
for column in missing_data.columns.values.tolist():
    print(column)
    print(missing_data[column].value_counts())
    print("")
    
    print(type(missing_data.columns.values.tolist())) # print the type of data

# Dealing with missing data by using the mean of the column:
# Step 1: Calculate the mean of the column:

avg_norm_loss = df['normalized-losses'].astype("float64").mean(axis=0)
print("Average of normalized-losses is", avg_norm_loss)

avg_stroke = df['stroke'].astype("float64").mean(axis = 0)
avg_bore = df['bore'].astype('float64').mean(axis = 0)
avg_horsepower = df['horsepower'].astype('float').mean(axis = 0)
avg_peakrpm = df['peak-rpm'].astype('float').mean(axis = 0)
print("average of stroke, bore, horsepower and peak-rpm are respectively:", avg_stroke, avg_bore, avg_horsepower, avg_peakrpm)

# Step 2: Replace "NaN" with the mean value in the columns
df['normalized-losses'].replace(np.nan, avg_norm_loss, inplace = True)
df['stroke'].replace(np.nan, avg_stroke, inplace = True)
df['bore'].replace(np.nan, avg_bore, inplace = True)
df['horsepower'].replace(np.nan, avg_horsepower, inplace = True)
df['peak-rpm'].replace(np.nan, avg_peakrpm, inplace = True)

print(df.head())

# Calculate and replace by the Frequency
# To see which values are present in a particular column, we can use the ".value_counts()" method:
df['num-of-doors'].value_counts()

# We can see that four doors are the most common type. We can also use the ".idxmax()" method to calculate the most common type automatically:
df['num-of-doors'].value_counts().idxmax()

# Replace all data by four:
df['num-of-doors'].replace(np.nan, "four", inplace = True)

# Drop all rows that do not have price data. In this case our goal is to predict the price of a car. If we have no price on a row, that row will not be that useful. 
df.dropna(subset = ["price"], axis = 0, how = 'any', inplace = True)

# reset index, because we droped two rows
df.reset_index(drop = True, inplace = True)
print(df.head())

# Correct data format with .dtypes and .astype
df.dtypes

# 'bore' and 'stroke' columns are numerical values that describe the engines, so we should expect them to be of the type 'float' or 'int'; however, they are shown as type 'object'.
# We can convert data types to proper format with .astype()

df[["bore", "stroke"]] = df[["bore", "stroke"]].astype('float')
df[["normalized-losses"]] = df[["normalized-losses"]].astype('int')
df[["price"]] = df[["price"]].astype('float')
df[["peak-rpm"]] = df[["peak-rpm"]].astype('float')

print(df.dtypes)

# Data Standardization
# Data is usually collected from different agencies in different formats. 
# Standardization is the process of transforming data into a common format, allowing the researcher to make the meaningful comparison.

# Convert mpg to L/100km by mathematical operation (235 divided by mpg)
df['city-L/100km'] = 235/ df["city-mpg"]

# check your transformed data 
print(df.head())

# transform mpg to L/100km in the column of "highway-mpg" and change the name of column to "highway-L/100km".
df['highway-mpg'] = 235/df["highway-mpg"]
df.rename(columns = {'highway-mpg':'highway-L/100km'}, inplace = True)

# Data Normalization
# Normalization is the process of transforming values of several variables into a similar range. 
# Typical normalizations include scaling the variable so the variable average is 0, scaling the variable so the variance is 1, or scaling the variable so the variable values range from 0 to 1.

# Example: to normalize the columns "length", "width" and "height" so their value ranges from 0 to 1
df['length'] = df["length"]/df["length"].max()
df['width'] = df['width']/df['width'].max()
df['height'] = df['height']/df['height'].max()
print(df[["length", "width", "height"]].head())

# Binning or creating "bins" for grouped analysis
# Binning is a process of transforming continuous numerical variables into discrete categorical 'bins' for grouped analysis.

# Convert data to correct format df.astype()
df["horsepower"] = df["horsepower"].astype('int64',copy = True)

# Let's plot the histogram of horsepower to see what the distribution of horsepower looks like.
%matplotlib inline
import matplotlib.pyplot as plt

plt.hist(df["horsepower"])

# set x/y labels and plot title
plt.xlabel("horsepower")
plt.ylabel("count")
plt.title("horsepower bins")

# Create bins with np.linspace(start_value, end_value, numbers_generated) function:
bins = np.linspace(min(df["horsepower"]), max(df["horsepower"]), 4)
print(bins)

# set the variable group_names = ['Low',...]
group_names = ['Low', 'Medium', 'High']

# Apply the function "cut" to determine what each value of df['horsepower'] belongs to.
df['horsepower_binned'] = pd.cut(df['horsepower'], bins, labels = group_names, include_lowest = True)
print(df[['horsepower', 'horsepower_binned']].head(50))

# Let's see the number of vehicles in each bin:
print(df['horsepower_binned'].value_counts())

# Plot the distribution of each bin with plt.bar()
%matplotlib inline
import matplotlib.pyplot as plt

plt.bar(group_names, df['horsepower_binned'].value_counts())

# set up labels
plt.xlabel('horsepower')
plt.ylabel('counts')
plt.title('horsepower bins')

# Bins Visualization with plt.hist(df['column'], bins = number)
import matplotlib.pyplot as plt
%matplotlib inline

plt.hist(df['horsepower'], bins = 3)
plt.xlabel("horsepower")
plt.ylabel("counts")
plt.title("horsepower bins")

# Indicator Variable (or Dummy Variable) with pd.get_dummies(df[ ])
# An indicator variable (or dummy variable) is a numerical variable used to label categories. They are called 'dummies' because the numbers themselves don't have inherent meaning.

# column "fuel-type" has 2 unique values: "gas" or "diesel".
# Regression doesn't understand words, only numbers. 
# To use this attribute in regression analysis, we convert "fuel-type" to indicator variables.

# Step1: Get the indicator variables and assign it to data frame "dummy_variable_1"
dummy_variable_1 = pd.get_dummies(df["fuel-type"])

# Change the column names for clarity
dummy_variable_1.rename(columns ={'gas': 'fuel-type-gas', 'diesel':'fuel-type-diesel'}, inplace = True)
print(dummy_variable_1.tail())

# Merge data frame "df" and "dummy_variable_1"
df = pd.concat([df, dummy_variable_1], axis = 1)
print(df)

# Drop original column "fuel-type" from "df"
df.drop("fuel-type", axis = 1, inplace = True)
print(df)

# Column "aspiration" can also be transformed to indicator variables
print(type(df['aspiration']))
df['aspiration'].value_counts()
dummy_variable_2 = pd.get_dummies(df['aspiration'])
dummy_variable_2.rename(columns = {'std':'aspiration-std', 'turbo': 'aspiration- turbo'}, inplace = True)
print(dummy_variable_2.tail())

# Merge the data frame and drop aspiration column
df = pd.concat([df, dummy_variable_2], axis = 1)
df.drop(['aspiration'], axis = 1, inplace = True)
print(df)

# Save the new csv file:
df.to_csv('clean_df_module2.csv')



