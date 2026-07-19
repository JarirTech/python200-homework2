import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import seaborn as sns
from scipy import stats
import os
os.makedirs("outputs", exist_ok=True)



#Part 1: Warmup Exercises-------------------

# --- scikit-learn API --- 

# -----scikit-learn Question 1 ---

years  = np.array([1, 2, 3, 5, 7, 10]).reshape(-1, 1)
salary = np.array([45000, 50000, 60000, 75000, 90000, 120000])

# y = mx+b
# salary = slope(m) * years(X) + intercept(b)
# create → fit → predict

# creating model
model = LinearRegression()

# Fit the model
model.fit(years, salary)

# predict

predct_4 = model.predict([[4]])[0] 
predct_8 = model.predict([[8]])[0]

print(' Scikit Question 1: \n')
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
print("Predicted salary for 4 years:", predct_4)
print("Predicted salary for 8 years:", predct_8)

#------------------------------------------------------------------------------

#----scikit-learn Question 2---------

x = np.array([10, 20, 30, 40, 50])

# shape of x ==> 1D
print(' Scikit Question 2:-------------------------------------- \n')
print("Original shape:",x.shape)

# reshape  ==>2D

x_reshaped = x.reshape(-1, 1)

print("Reshaped shape:", x_reshaped.shape)

# scikit-learn need X to be 2D  so that 's why we reshaped it to 2D 
# because rows represent samples and columns represent features.
#--------------------------------------------------------------------------------------------------


#---scikit-learn Question 3----------

X_clusters, _ = make_blobs(n_samples=120, centers=3, cluster_std=0.8, random_state=7)

# creating K-Means model
kmeans = KMeans(n_clusters=3, random_state=42)


# fit the model to the data
kmeans.fit(X_clusters)

labels = kmeans.predict(X_clusters)
print(' Scikit Question 3:-------------------------------------- \n')
print("Cluster Centers:\n", kmeans.cluster_centers_)
print("Points in each cluster:", np.bincount(labels))

# Plot clusters
plt.figure(figsize=(8,6))
plt.scatter( X_clusters[:,0], X_clusters[:,1], c=labels, cmap="viridis")

plt.scatter( kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], marker="X", s=250,
    c="black")

plt.title("KMeans Clusters")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.savefig("outputs/kmeans_clusters.png")
plt.close()

#-------------------------------------------------------------------------------------------------

#---Linear Regression---

np.random.seed(42)
num_patients = 100
age    = np.random.randint(20, 65, num_patients).astype(float) # 100 random ages
smoker = np.random.randint(0, 2, num_patients).astype(float)
cost   = 200 * age + 15000 * smoker + np.random.normal(0, 3000, num_patients)

# ---Linear Regression Question 1---
#print('age: ', age)
print('----- Linear Regression Question 1 :-------------------------------------- \n')
plt.figure(figsize=(8,6))
plt.scatter(age, cost, c=smoker, cmap="coolwarm")
plt.title("Medical Cost vs Age")
plt.xlabel("Age")
plt.ylabel("Medical Cost")
plt.savefig("outputs/cost_vs_age.png")
plt.close()

# Comment:
# Yes there is 2  groups  with different colors.
# Smokers have much higher costs than non-smokers. Smoking cost more medical expanses

#-----------------------------------------------------------

#Linear Regression Question 2
print('----- Linear Regression Question 2 :-------------------------------------- \n')
X = age.reshape(-1, 1)
y = cost

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)

print("X_train shape:", X_train.shape)#80 patients and one feature (age)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

#-----------------------------------------------------------

#Linear Regression Question 3
print('----- Linear Regression Question 3 :-------------------------------------- \n')

model_age = LinearRegression()
model_age.fit(X_train, y_train)

print("Slope:", model_age.coef_[0])
print("Intercept:", model_age.intercept_)

y_pred = model_age.predict(X_test)

rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
r2 = model_age.score(X_test, y_test)

print("RMSE:", rmse)
print("R² on test set:", r2)

# Comment:
# The slope means each extra year of age increases the medical cost 

#-----------------------------------------------------------

#Linear Regression Question 4
print('----- Linear Regression Question 4 :-------------------------------------- \n')

X_full = np.column_stack([age, smoker])
X_train2, X_test2, y_train2, y_test2 = train_test_split( X_full, y, test_size=0.2, random_state=42)

model_full = LinearRegression()
model_full.fit(X_train2, y_train2)

r2_full = model_full.score(X_test2, y_test2)

print("R² using age only:", r2)
print("R² using age + smoker:", r2_full)

print("age coefficient:   ", model_full.coef_[0])
print("smoker coefficient:", model_full.coef_[1])

# Comment:
# Smoker coefficient represents how much more a smoker
# is predicted to pay yearly compared to a non-smoker

#-----------------------------------------------------------

#Linear Regression Question 5
print('----- Linear Regression Question 5:-------------------------------------- \n')

y_pred2 = model_full.predict(X_test2)

plt.figure(figsize=(8,6))
plt.scatter(y_pred2, y_test2)

# Diagonal line
min_val = min(y_pred2.min(), y_test2.min())
max_val = max(y_pred2.max(), y_test2.max())

plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")

plt.title("Predicted vs Actual")
plt.xlabel("Predicted Cost")
plt.ylabel("Actual Cost")

plt.savefig("outputs/predicted_vs_actual.png")
plt.close()

# Comment:
# Above diagonal means actual cost is higher than predicted
# Below diagonal => model predicted too high