# ======= Part 1: Warmup Exercises===========


#**************  Pandas Review *****************
#----------Pandas Question 1-----------


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from scipy import stats
from scipy.stats import pearsonr
import seaborn as sns
print('Pandas Question 1')

data = {
    "name":   ["Alice", "Bob", "Carol", "David", "Eve"],
    "grade":  [85, 72, 90, 68, 95],
    "city":   ["Boston", "Austin", "Boston", "Denver", "Austin"],
    "passed": [True, True, True, False, True]
}
df = pd.DataFrame(data)

# print the first three rows, the shape, and the data types of each column.
# Print each result with a label (e.g. print(f"Num Rows: {len(df)}")).

print("First three rows: \n", df.head(3))
print("Shape: ", df.shape)
print("Data Types: ", df.dtypes)
#====================================================================================
#----------Pandas Question 2-----------
print('Pandas Question 2')
# Filter the DataFrame to include only students who passed (passed == True) and have 
# a grade greater than 80. Print the resulting DataFrame.

filtered_df = df[(df["passed"] == True) & (df["grade"] > 80)]
print("Filtered DataFrame: \n", filtered_df)
#====================================================================================
#----------Pandas Question 3-----------
print('Pandas Question 3')
# Add a new column called "grade_curved" that adds 5 points to each student's grade. 
# Print the updated DataFrame (all columns, all rows)
df["grade_curved"] = df["grade"] + 5
print("Updated DataFrame: \n", df)
#---------------------------------------------------------------------------------
# Question 4

#Add a new column called "name_upper" that contains each student's name in uppercase, 
# using the .str accessor. Print the "name" and "name_upper" columns together.

print('Pandas Question 4')
df['name_upper']= df['name'].str.upper()
print(f" name: \n{df['name']} \nand \nname_upper: \n{df['name_upper']}")

#--------------------------------------------------------------------------------
# Question 5
print('Pandas Question 5')
#Group the DataFrame by "city" and compute the mean grade for each city. Print the result.

grouped_df = df.groupby('city')['grade'].mean()
print(f"grouped_df:\n{grouped_df}")

#---------------------------------------------------------------------------------------

# Question 6:
#Replace the value "Austin" in the "city" column with "Houston". Print the "name" and
#  "city" columns to confirm the change
print('Pandas Question 6')                      
df['city'] = df['city'].str.replace("Austin", "Houston")
print(df[['name', 'city']])

#------------------------------------------------------------------------------------------
# Question 7:

print('Pandas Question 7')
#Sort the DataFrame by "grade" in descending order and print the top 3 rows.
sorted_df = df.sort_values(by= 'grade' , ascending = False)

print(sorted_df.head(3))


#----------------------------------------------------------------------------------------

# NumPy Review
# Question 1
print('NumPy Question 1')
# Create a 1D NumPy array from the list [10, 20, 30, 40, 50]. Print its shape, dtype, and ndim.
num_array = np.array([10, 20, 30, 40, 50])
print("Shape:", num_array.shape)
print("Data type:", num_array.dtype)
print("Number of dimensions:", num_array.ndim)

#---------------------------------------------------------------------------------------------------

# NumPy Question 2
print('NumPy Question 2')
# Create the following 2D array and print its shape and size (total number of elements).

arr = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])

print("Shape:", arr.shape)
print("Size:", arr.size)
#---------------------------------------------------------------------------------------
# NumPy Question 3
print('NumPy Question 3')

#Using the 2D array from Q2, slice out the top-left 2x2 block and print it. 
# The expected result is [[1, 2], [4, 5]]


sliced_array = arr[0:2, 0:2]

print(sliced_array)

#-------------------------------------------------------------------------------------------
#NumPy Question 4
print('NumPy Question 4')
# Create a 3x4 array of zeros using a built-in command. Then create a 2x5 array of ones 
# using a built-in command. Print both.

#3x4 array of zeros
zeros_array = np.zeros((3, 4))

# 2x5 array of ones
ones_array = np.ones((2, 5))

# Print both
print("Zeros array:\n", zeros_array)
print("\nOnes array:\n", ones_array)

#------------------------------------------------------------------------------------
#NumPy Question 5
print('NumPy Question 5')
# Create an array using np.arange(0, 50, 5). First, think about what you expect it to look like.
#  Then, print the array, its shape, mean, sum, and standard deviation.
aranged_array = np.arange(0, 50, 5)

# Print results
print("Array:", aranged_array)
print("Shape:", aranged_array.shape)
print("Mean:", aranged_array.mean())
print("Sum:", aranged_array.sum())
print("Standard Deviation:", aranged_array.std())
#-------------------------------------------------------------------------------------------------

# NumPy Question 6
print('NumPy Question 6')
#Generate an array of 200 random values drawn from a normal distribution with mean 0 and standard deviation
#1 (use np.random.normal()). Print the mean and standard deviation of the result.
arr = np.random.normal(loc=0, scale=1, size=200)

print("Mean:", arr.mean())
print("Standard Deviation:", arr.std())
#------------------------------------------------------------------------------------------------------------
# Matplotlib

#Matplotlib Question 1
print('Matplotlib Question 1')
#Plot the following data as a line plot. Add a title "Squares", x-axis label "x", and y-axis label "y".

x = [0, 1, 2, 3, 4, 5]
y = [0, 1, 4, 9, 16, 25]
plt.plot(x, y, marker= 'o')

#title and labels
plt.title("Squares")
plt.xlabel("x")
plt.ylabel("y")

# Show the plot
plt.show()

# Matplotlib Question 2
print('Matplotlib Question 2')
#Create a bar plot for the following subject scores. Add a title "Subject Scores" and label both axes.
subjects = ["Math", "Science", "English", "History"]
scores   = [88, 92, 75, 83]


# Create the bar plot
plt.bar(subjects, scores)

# title and axis labels
plt.title("Subject Scores")
plt.xlabel("Subjects")
plt.ylabel("Scores")

# Displaying the plot
plt.show()

#Matplotlib Question 3
print('Matplotlib Question 3')
#Plot the two datasets below as a scatter plot on the same figure. Use different colors for each,
#  add a legend, and label both axes.

x1, y1 = [1, 2, 3, 4, 5], [2, 4, 5, 4, 5]
x2, y2 = [1, 2, 3, 4, 5], [5, 4, 3, 2, 1]

plt.scatter(x1, y1, color='blue', label='Dataset 1')
plt.scatter(x2, y2, color='red', label='Dataset 2')

# labels 
plt.xlabel("x")
plt.ylabel("y")
plt.title("Scatter Plot of Two Datasets")
plt.legend()

# Showing the plot
plt.show()

#Matplotlib Question 4
print('Matplotlib Question 4')
#Use plt.subplots() to create a figure with 1 row and 2 subplots side by side.
#  In the left subplot, plot x vs y from Q1 as a line. In the right subplot,
#  plot the subjects and scores from Q2 as a bar plot. Add a title to each subplot and 
# call plt.tight_layout() before showing.

# Creating subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Left subplot: line plot
ax1.plot(x, y, marker='o')
ax1.set_title("Squares")
ax1.set_xlabel("x")
ax1.set_ylabel("y")

# Right subplot: bar plot
ax2.bar(subjects, scores)
ax2.set_title("Subject Scores")
ax2.set_xlabel("Subjects")
ax2.set_ylabel("Scores")

# Adjust layout
plt.tight_layout()

# Showing the figure

plt.show()


#-------------------------------------------------------------------
# Descriptive Stats

#Descriptive Stats Question 1
print('Descriptive Stats Question 1')
# Convert to NumPy array

data = [12, 15, 14, 10, 18, 22, 13, 16, 14, 15]
arr = np.array(data)

# Compute statistics
mean = np.mean(arr)
median = np.median(arr)
variance = np.var(arr)
std_dev = np.std(arr)

# Print results with labels
print("Mean:", mean)
print("Median:", median)
print("Variance:", variance)
print("Standard Deviation:", std_dev)


#Descriptive Stats Question 2
print('Descriptive Stats Question 2')
#bGenerate 500 random values from a normal distribution with mean 65 and standard deviation 10
#  (use np.random.normal(65, 10, 500)). Plot a histogram with 20 bins. Add a title "Distribution 
# of Scores" and label both axes.

data = np.random.normal(65, 10, 500)

# histogram
plt.hist(data, bins = 18, edgecolor='black')

# labels and title
plt.title("Distribution of Scores")
plt.xlabel("Score")
plt.ylabel("Frequency")

# Displaying plot
plt.show()

# Descriptive Stats Question 3
print('Descriptive Stats Question 3')
# Create a boxplot comparing the two groups below. Label each box ("Group A" and "Group B") 
# and add a title "Score Comparison".
group_a = [55, 60, 63, 70, 68, 62, 58, 65]
group_b = [75, 80, 78, 90, 85, 79, 82, 88]

# Creating boxplot
#plt.boxplot([group_a, group_b], labels=["Group A", "Group B"])
plt.boxplot([group_a, group_b], tick_labels=["Group A", "Group B"])

# Adding title
plt.title("Score Comparison")

# Showing the plot
plt.show()

# Descriptive Stats Question 4
print('Descriptive Stats Question 4')
# You are given two datasets: one normally distributed and one 'exponential' distribution.



normal_data = np.random.normal(50, 5, 200)
skewed_data = np.random.exponential(10, 200)

# Creating boxplot
plt.boxplot([normal_data, skewed_data], tick_labels=["Normal", "Exponential"])

# Adding title
plt.title("Distribution Comparison")

# Showing the plot
plt.show()
### The exponential data is more skewed, so median works better for it, while the normal
### data is balanced so mean is fine.

# Descriptive Stats Question 5
print('Descriptive Stats Question 5')
# Print the mean, median, and mode of the following:

data1 = [10, 12, 12, 16, 18]

data2 = [10, 12, 12, 16, 150]
# Why are the median and mean so different for data2? Add your answer as a comment in the code.

# Data1 stats
print("Data1 Mean:", np.mean(data1))
print("Data1 Median:", np.median(data1))
print("Data1 Mode:", stats.mode(data1, keepdims=True)[0][0])

# Data2 stats
print("Data2 Mean:", np.mean(data2))
print("Data2 Median:", np.median(data2))
print("Data2 Mode:", stats.mode(data2, keepdims=True)[0][0])

### The mean is much higher in data2 because the outlier 
### (150), while the median stays more stable.

#--------------------------------------------------------------------------------------

# Hypothesis Testing Review
print('Hypothesis Testing Review Q1')

#t-test
t_stat, p_value = stats.ttest_ind(group_a, group_b)

# Print results
print("t-statistic:", t_stat)
print("p-value:", p_value)

# Hypothesis Question 2
print('Hypothesis Question 2')
#Using the p-value from Q1, write an if/else statement that prints whether the 
# result is statistically significant at alpha = 0.05.
# Checking significance
alpha = 0.05

if p_value < alpha:
    print("The result is  significant.")
else:
    print("The result is not significant.")

# Hypothesis Question 3
print('Hypothesis Question 3')
#Run a paired t-test on the before/after scores below (the same students 
# measured twice). Print the t-statistic and p-value.

before = [60, 65, 70, 58, 62, 67, 63, 66]
after  = [68, 70, 76, 65, 69, 72, 70, 71]

# Perform paired t-test
t_stat, p_value = stats.ttest_rel(before, after)

# Print results
print("t-statistic:", t_stat)
print("p-value:", p_value)

#Hypothesis Question 4
print('Hypothesis Question 4')
#Run a one-sample t-test to check whether the mean of scores is significantly different
#  from a national benchmark of 70. Print the t-statistic and p-value.
scores = [72, 68, 75, 70, 69, 74, 71, 73]

# test against benchmark mean = 70
t_stat, p_value = stats.ttest_1samp(scores, 70)

# Print results
print("t-statistic:", t_stat)
print("p-value:", p_value)

# Hypothesis Question 5
print('Hypothesis Question 5')
# Re-run the test from Q1 as a one-tailed test to check whether group_a scores are 
# less than group_b scores. Print the resulting p-value. Use the alternative parameter.

group_a = [72, 68, 75, 70, 69, 73, 71, 74]
group_b = [80, 85, 78, 83, 82, 86, 79, 84]

result = stats.ttest_ind(group_a, group_b, alternative="less")

p_value = result.pvalue

print("One-tailed p-value (group_a < group_b):", p_value)

# Hypothesis Question 6
print('Hypothesis Question 6')
# Write a plain-language conclusion for the result of Q1 (do not just say "reject the null hypothesis").
#  Format it as a print() statement. Your conclusion should mention the direction of the difference and 
# whether it is likely due to chance.
print("Group B scored higher than Group A, and the difference is so large that it "
"is very unlikely to be due to chance.")
#---------------------------------------------------------------------------------------------------------

# Correlation Review
# Correlation Question 1
print('Correlation Question 1')
#Compute the Pearson correlation between x and y below using np.corrcoef(). Print the full correlation 
# matrix, then print just the correlation coefficient (the value at position [0, 1]).
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]
corr_matrix = np.corrcoef(x, y)
print("Correlation matrix:")
print(corr_matrix)

corr_value = corr_matrix[0, 1]
print("Correlation coefficient:", corr_value)

# I expect the correlation to be 1.0 because y is a perfect linear multiple of x.

# Correlation Question 2
print('Correlation Question 2')
#Use pearsonr() from scipy.stats to compute the correlation between x and y below. Print both the correlation coefficient and the p-value.


x = [1,  2,  3,  4,  5,  6,  7,  8,  9, 10]
y = [10, 9,  7,  8,  6,  5,  3,  4,  2,  1]
corr, p_value = pearsonr(x, y)

print("Correlation coefficient:", corr)
print("p-value:", p_value)

#Correlation Question 3
print('Correlation Question 3')
# Create the following DataFrame and use df.corr() to compute the correlation matrix. Print the result.
people = {
    "height": [160, 165, 170, 175, 180],
    "weight": [55,  60,  65,  72,  80],
    "age":    [25,  30,  22,  35,  28]
}
df = pd.DataFrame(people)

corr_matrix = df.corr()
print(corr_matrix)

# Correlation Question 4
print('Correlation Question 4')
# Create a scatter plot of x and y below, which have
#  a negative relationship. Add a title "Negative Correlation" and label both axes.
x = [10, 20, 30, 40, 50]
y = [90, 75, 60, 45, 30]
plt.scatter(x, y)
plt.title("Negative Correlation")
plt.xlabel("X values")
plt.ylabel("Y values")

plt.show()
#Correlation Question 5
print('Correlation Question 5')
#Using the correlation matrix from Q3, create a heatmap with sns.heatmap(). Pass annot=True so the correlation values appear in each cell, and add a title "Correlation Heatmap".

df = pd.DataFrame(people)

corr_matrix = df.corr()

sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

#-----------------------------------------------------------------------------------------

# Pipelines
# Pipeline Question 1
print('Pipeline Question 1')
arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])

# 1. Create series
def create_series(arr):
    return pd.Series(arr, name="values")


# 2. Clean data
def clean_data(series):
    return series.dropna()


# 3. Summarize data
def summarize_data(series):
    return {
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
        "mode": series.mode()[0]
    }


# 4. Pipeline function
def data_pipeline(arr):
    series = create_series(arr)
    cleaned = clean_data(series)
    summary = summarize_data(cleaned)
    return summary


# Run pipeline
result = data_pipeline(arr)

# Print results
for key, value in result.items():
    print(key, ":", value)