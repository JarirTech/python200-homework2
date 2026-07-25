import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

os.makedirs("outputs", exist_ok=True)
print('Task1:\n')
## fields are seperated by semicolones;
#---Task 1: Load and Explore -----
df = pd.read_csv('student_performance_math.csv', sep=";")

print(df.head())
print(df.tail())
print(df.shape) #(395, 18)
df.info()
#  histogram of G3 with 21 bins 
plt.figure(figsize=(8,6))
plt.hist(df["G3"], bins=21, edgecolor="black")
plt.title("Distribution of Final Math Grades")
plt.xlabel("Final Grade (G3)")
plt.ylabel("Count")
plt.savefig("outputs/g3_distribution.png")
plt.close()

print('Task2:----------------------------------------------------------------\n')
print('shape before removing Zeros: ', df.shape)
filtred_df = df[df['G3']>0].copy()
print('filtred df shape: ', filtred_df.shape)#(357, 18)

#Then convert the yes/no columns to 1/0 and the sex column to 0/1.
# determining columns with yes/no:

yes_cols = ["schoolsup", "internet", "higher", "activities"]
for col in yes_cols:
    df[col] = df[col].map({"yes":1, "no":0})
    filtred_df[col] = filtred_df[col].map({"yes":1, "no":0})

df["sex"] = df["sex"].map({"F":0, "M":1})
filtred_df["sex"] = filtred_df["sex"].map({"F":0, "M":1})

#Compute the Pearson correlation between absences and G3 on both 
# the original dataset and the filtered one

corr_original = df['absences'].corr(df['G3'])
corr_filtred_df = filtred_df['absences'].corr(filtred_df['G3'])
print('correlation of original df is: ', corr_original)
print('correlation of filtered df is: ', corr_filtred_df)
# Students with G3 = 0 were removed because many of them did not
# actually complete the course or the final exam. Keeping these rows
# would make the model learn from students who were not really graded,
# which could reduce the accuracy of the predictions.
#
# In the original dataset, students with G3 = 0 made absences look like
# a very weak predictor because many of them had different attendance
# patterns but all received a final grade of zero. After removing these
# rows, the true relationship became clearer: students with more
# absences generally tended to earn lower final grades.

print('Task3:----------------------------------------------------------------\n')
numeric_features = ["age","Medu","Fedu","traveltime","studytime","failures","absences","freetime","goout","Walc"]
corrs={}
for col in numeric_features:
    corrs[col]=filtred_df[col].corr(filtred_df["G3"])
for n,v in sorted(corrs.items(), key=lambda x:x[1]):
    print(f"{n:10s}: {v:+.3f}")
# Plot 1: Failures vs G3
plt.figure(figsize=(8,6))
plt.scatter(filtred_df["failures"],filtred_df["G3"],alpha=0.7)
plt.title("Failures vs Final Grade")
plt.xlabel("Past Failures")
plt.ylabel("G3")
plt.savefig("outputs/failures_vs_g3.png")
plt.close()
# More failures generally means lower grades.

# Plot 2: Studytime vs G3
plt.figure(figsize=(8,6))
plt.scatter(filtred_df["studytime"],filtred_df["G3"],alpha=0.7)
plt.title("Study Time vs Final Grade")
plt.xlabel("Study Time")
plt.ylabel("G3")
plt.savefig("outputs/studytime_vs_g3.png")
plt.close()

# More study time tends to give better grades

print('Task4:----------------------------------------------------------------\n')
X=filtred_df[["failures"]].values
y=filtred_df["G3"].values
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model=LinearRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
rmse=np.sqrt(np.mean((y_pred-y_test)**2))
r2=model.score(X_test,y_test)
print("\n--Baseline Model --")
print("Slope:",model.coef_[0])
print("RMSE:",rmse)
print("R²:",r2)

# comment 
# each extra past failure lowers the predicted final grade by about 1.43 points, and the model 
# misses by about 3 points on a 0-20 scale.
# R² is low (0.09), so the model is worse than expected and failures alone do not predict
#  grades very well.

print('Task5:----------------------------------------------------------------\n')
feature_cols=["age","Medu","Fedu","traveltime","studytime","failures","absences","freetime","goout","Walc","schoolsup","internet","higher","activities","sex"]
X=filtred_df[feature_cols].values
y=filtred_df["G3"].values
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model_full=LinearRegression()
model_full.fit(X_train,y_train)
train_r2=model_full.score(X_train,y_train)
test_r2=model_full.score(X_test,y_test)
y_pred=model_full.predict(X_test)
rmse=np.sqrt(np.mean((y_pred-y_test)**2))
print("Filtered dataset size:", len(filtred_df))
print("Test set size:", len(y_test))
print("\n ---Full Model ---")
print("Train R²:",train_r2)
print("Test R² :",test_r2)
print("RMSE    :",rmse)
print("\nCoefficients:")
for name,coef in zip(feature_cols,model_full.coef_):
    print(f"{name:12s}: {coef:+.3f}")

# Find the largest positive and negative coefficients
coef_pairs = list(zip(feature_cols, model_full.coef_))

largest_positive = max(coef_pairs, key=lambda x: x[1])
largest_negative = min(coef_pairs, key=lambda x: x[1])

print("\nLargest positive coefficient:")
print(f"{largest_positive[0]}: {largest_positive[1]:+.3f}")

print("Largest negative coefficient:")
print(f"{largest_negative[0]}: {largest_negative[1]:+.3f}")

# Failures has the biggest negative effect, so more past failures usually mean lower final grades.
# Internet, higher education goal, and studytime help grades, while schoolsup is negative because 
# students needing support may already be struggling.
# Train R² and Test R² are close, so the model is not overfitting, but both scores are still low.
# RMSE is about 2.86, so predictions are usually off by almost 3 grade points on a 0-20 scale.


print('Task6:----------------------------------------------------------------\n')
plt.figure(figsize=(8,6))
plt.scatter(y_pred,y_test,alpha=0.7)
min_val=min(y_pred.min(),y_test.min())
max_val=max(y_pred.max(),y_test.max())
plt.plot([min_val,max_val],[min_val,max_val],"--")
plt.title("Predicted vs Actual (Full Model)")
plt.xlabel("Predicted Grade")
plt.ylabel("Actual Grade")
plt.savefig("outputs/predicted_vs_actual.png")
plt.close()

# Most points are around the diagonal line, so the model gives reasonable predictions, but there is still
#  some error.
# The model seems to struggle more with very high and very low grades; points above the line mean actual 
# grade was higher than predicted, and points below mean predicted grade was too high.

#Neglected Feature: The Power of G1


feature_cols_g1=feature_cols+["G1"]
X=filtred_df[feature_cols_g1].values
y=filtred_df["G3"].values
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model_g1=LinearRegression()
model_g1.fit(X_train,y_train)
r2_g1=model_g1.score(X_test,y_test)
print("\n=== Full Model + G1 ===")
print(f"Original Full Model Test R²: {test_r2:.3f}")
print(f"Full Model + G1 Test R²:     {r2_g1:.3f}")
print(f"Improvement:                 {r2_g1-test_r2:.3f}")

# Adding G1 greatly improves the model because first-period grades are
# strongly related to final grades.
# A high R² does not mean G1 causes G3; it simply provides strong predictive
# information. This model is useful after the first grading period, but it
# cannot identify struggling students before G1 is available.