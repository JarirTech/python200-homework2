import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import os
os.makedirs("outputs", exist_ok=True)

iris = load_iris(as_frame=True)
X = iris.data
y = iris.target

#------Part 1: Warmup Exercises---------------------------------------------------

#----------Preprocessing----------------------------------------------------------
#---------Preprocessing Question 1------------------------------------------------
print('---------Preprocessing Question 1------------------------------------------------')
#Split X and y into training and test sets using an 80/20 split with stratify=y and random_state=42.
# create --->  fit --->  predict---> evaluate 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, stratify=y, random_state = 42)
# Print the shapes of all four arrays.
# print(X_train.head())
# print(y_train.head())
# print(X_test.head())
# print(y_test.head())
print("X_train shape: ", X_train.shape)
print("y_train shape: ", y_train.shape)
print("X_test shape: ", X_test.shape)
print("y_test shape: ", y_test.shape)
#---------------------------------------------------------------------------------------
#---------Preprocessing Question 2------------------------------------------------
print('---------Preprocessing Question 2------------------------------------------------')
#Fit a StandardScaler on X_train and use it to transform both X_train and X_test

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Print the mean of each column in X_train_scaled -- 
print("Means of scaled X_train columns:")
print(X_train_scaled.mean(axis=0))

#comment: scaling x train only to avaoid data leakage
#-----KNN----------------------------------------------------------------------
#---------KNN Question 1------------------------------------------------
print('---------KNN Question 1------------------------------------------------')

#Build a KNeighborsClassifier with n_neighbors=5, fit it on the unscaled training data (X_train),
#  and predict on the test set.
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

#Print the accuracy score and the full classification report.
print("Accuracy:", accuracy_score(y_test, y_pred))
print('classification report:\n')
print(classification_report(y_test, y_pred))

#---------KNN Question 2------------------------------------------------
print('---------KNN Question 2------------------------------------------------')
#Repeat KNN Question 1 using the scaled data (X_train_scaled, X_test_scaled). 
scaled_knn = KNeighborsClassifier(n_neighbors=5)
scaled_knn.fit(X_train_scaled, y_train)

y_pred_scaled = scaled_knn.predict(X_test_scaled)
print("Accuracy (scaled):", accuracy_score(y_test, y_pred_scaled))

# Scaling makes no noticeable difference on the Iris dataset because all four
# features are already measured in centimeters and have similar ranges.
# Since no feature dominates the distance calculation, KNN performs about
# the same with or without scaling.

#---------KNN Question 3------------------------------------------------
print('---------KNN Question 3------------------------------------------------')
#Using cross_val_score with cv=5, evaluate the k=5 KNN model on the unscaled training data. 
knn = KNeighborsClassifier(n_neighbors=5)
cv_scores = cross_val_score(knn, X_train, y_train, cv=5)


#Print each fold score, the mean, and the standard deviation

# print(cv_scores)
# print(f"Mean: {cv_scores.mean():.3f}")
# print(f"Std: {cv_scores.std():.3f}")

for i, score in enumerate(cv_scores, start=1):
    print(f"Fold {i}: {score:.3f}")

print(f"Mean: {cv_scores.mean():.3f}")
print(f"Std : {cv_scores.std():.3f}")

# Cross-validation is more trustworthy than one train/test split
# because every sample is used for testing once.


##---------KNN Question 4------------------------------------------------
print('---------KNN Question 4------------------------------------------------')
#Loop over k values [1, 3, 5, 7, 9, 11, 13, 15]. For each, compute 5-fold cross-validation 
# accuracy on the unscaled training data 

k_values = [1,3,5,7,9,11,13,15]

for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    #print k and the mean CV score
    print(f"k={k}, mean CV={cv_scores.mean():.4f}")

# I would choose the value of k with the highest mean cross-validation accuracy
# because it performs best across multiple train/test splits and is more likely
# to generalize well to new data.

#------Classifier Evaluation --------------------------------------------------------------
#---------Classifier Evaluation Question 1------------------------------------------------
print('-------Classifier Evaluation Question 1------------------------------------------------')

#Using your predictions from KNN Question 1, create a confusion matrix and display it with 
# ConfusionMatrixDisplay, passing display_labels=iris.target_names.

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=iris.target_names
)

disp.plot()
plt.title("KNN Confusion Matrix")
plt.savefig("outputs/knn_confusion_matrix.png")
plt.close()

# The model does not confuse any species.
# Every test sample was classified correctly.

#------The sklearn API: Decision Trees-------------------------------------------
#------The sklearn API: Decision TreesQuestion 1------------------------------------------------
print('-----The sklearn API: Decision Trees Question 1------------------------------------------------')
tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X_train, y_train)

tree_pred = tree.predict(X_test)

print("Accuracy:", accuracy_score(y_test, tree_pred))
print(classification_report(y_test, tree_pred))

# The Decision Tree achieved slightly lower accuracy than the KNN model
# (96.7% compared with the higher KNN accuracy).
# Scaling would not change Decision Tree performance because trees make
# splits using feature thresholds rather than distance calculations.

#-------------------------------------------------------------------------------
#------Logistic Regression and Regularization-------------------------------------------


#------Logistic Regression Question 1------------------------------------------------
print('-----Logistic Regression Question 1------------------------------------------------')

for c in [0.01, 1.0, 100]:
    log_reg = OneVsRestClassifier(
        LogisticRegression(
            C=c,
            solver="liblinear",
            max_iter=1000
        )
    )

    log_reg.fit(X_train_scaled, y_train)

    # Sum of absolute coefficients across all binary classifiers
    coef_size = sum(
        np.abs(est.coef_).sum()
        for est in log_reg.estimators_
    )

    print(f"C={c}, Total coefficient size={coef_size:.4f}")

# As C increases, regularization becomes weaker, allowing larger coefficients.
# Smaller C applies stronger regularization, shrinking the coefficients.
#-------------------------------------------------------------------------------
#------PCA -------------------------------------------
#------PCA Question 1------------------------------------------------
print('-----PCA Question 1------------------------------------------------')

digits = load_digits()

X_digits = digits.data
y_digits = digits.target
images = digits.images


print("X_digits shape:", X_digits.shape)
print("images shape:", images.shape)

fig, axes = plt.subplots(1, 10, figsize=(15, 3))

for digit in range(10):
    idx = np.where(y_digits == digit)[0][0]
    axes[digit].imshow(images[idx], cmap="gray_r")
    axes[digit].set_title(str(digit))
    axes[digit].axis("off")

plt.tight_layout()
plt.savefig("outputs/sample_digits.png")
plt.close()

#--------------------------------------------------------------------------------
#------PCA Question 2------------------------------------------------
print('-----PCA Question 2------------------------------------------------')
pca = PCA()
pca.fit(X_digits)

scores = pca.transform(X_digits)

plt.figure(figsize=(8,6))
scatter = plt.scatter(
    scores[:,0],
    scores[:,1],
    c=y_digits,
    cmap="tab10",
    s=10
)

plt.colorbar(scatter, label="Digit")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA 2D Projection")
plt.savefig("outputs/pca_2d_projection.png")
plt.close()

# yes same digits form clusters.

#-----------------------------------------------------
#------PCA Question 3------------------------------------------------
print('-----PCA Question 3------------------------------------------------')
cum_var = np.cumsum(pca.explained_variance_ratio_)

plt.figure(figsize=(8,6))
plt.plot(range(1, len(cum_var)+1), cum_var, marker="o")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Variance Explained")
plt.grid(True)
plt.savefig("outputs/pca_variance_explained.png")
plt.close()

n80 = np.argmax(cum_var >= 0.80) + 1
print("Components for 80% variance:", n80)

#-----------------------------------------------------
#------PCA Question 4------------------------------------------------
print('-----PCA Question 4------------------------------------------------')

def reconstruct_digit(sample_idx, scores, pca, n_components):
    reconstruction = pca.mean_.copy()

    for i in range(n_components):
        reconstruction = reconstruction + scores[sample_idx, i] * pca.components_[i]

    return reconstruction.reshape(8, 8)

n_list = [2, 5, 15, 40]

fig, axes = plt.subplots(len(n_list)+1, 5, figsize=(10,10))

# Original row
for col in range(5):
    axes[0, col].imshow(images[col], cmap="gray_r")
    #axes[0, col].set_title(f"Orig {col}")
    axes[0, col].set_title(f"Digit {col}")
    axes[0, 0].set_ylabel("Original", fontsize=12)
    axes[0, col].axis("off")

# Reconstruction rows
for row, n in enumerate(n_list, start=1):
    for col in range(5):
        recon = reconstruct_digit(col, scores, pca, n)
        axes[row, col].imshow(recon, cmap="gray_r")
        #axes[row, col].set_title(f"n={n}")
        if col == 0:
            axes[row, col].set_ylabel(f"n={n}", fontsize=12)
        axes[row, col].axis("off")

plt.tight_layout()
plt.savefig("outputs/pca_reconstructions.png")
plt.close()

# The digits become clearly recognizable around n=15 components.
# At n=2 the images are blurry, and at n=5 they improve but still lose detail.
# By n=15 most digits are easy to identify, while n=40 looks very close
# to the original images.

# Yes, this generally matches the variance curve because recognition improves
# as more important components are added, and after that the gains become smaller.
