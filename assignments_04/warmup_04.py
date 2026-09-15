# Part 1: Warmup Exercises


import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    RocCurveDisplay,
    classification_report,
      f1_score,
)
import joblib

os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Synthetic dataset — binary classification, two informative features
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=4,
    n_redundant=2,
    random_state=42,
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ====== ROC and AUC*************************************************
#=====================================================================
#***************ROC Question 1***************
print('***************ROC Question 1***************')

# Logistic Regression on the raw data
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)

# KNN on scaled data

scaler = StandardScaler()

# Scale the training data
X_train_scaled = scaler.fit_transform(X_train)

# Use the same scaling on the test data
X_test_scaled = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# Get predicted probabilities

y_probs_lr = lr.predict_proba(X_test)[:, 1]
y_probs_knn = knn.predict_proba(X_test_scaled)[:, 1]

# AUC
auc_lr = roc_auc_score(y_test, y_probs_lr)
auc_knn = roc_auc_score(y_test, y_probs_knn)

print("Logistic Regression AUC:", auc_lr)
print("KNN AUC:", auc_knn)

# KNN has the higher AUC (0.9394) compared to Logistic Regression (0.7060).
# This means KNN does a better job separating the positive and negative classes.
# Since AUC looks at all possible thresholds, KNN performs better overall, not just at one cutoff value.

#=====================================================================

#********ROC Question 2***************
print('***************ROC Question 2***************')

plt.figure(figsize=(7,6))

RocCurveDisplay.from_predictions(
    y_test,
    y_probs_lr,
    name=f"Logistic Regression (AUC={auc_lr:.3f})"
)

RocCurveDisplay.from_predictions(
    y_test,
    y_probs_knn,
    name=f"KNN (AUC={auc_knn:.3f})"
)

plt.plot([0,1],[0,1],"k--",label="Random Classifier")
plt.title("ROC Curve Comparison")
plt.savefig("outputs/roc_comparison.png")
plt.close()

# Find ROC values
fpr_lr, tpr_lr, thresh_lr = roc_curve(y_test, y_probs_lr)
fpr_knn, tpr_knn, thresh_knn = roc_curve(y_test, y_probs_knn)

idx_lr = np.argmin(np.abs(tpr_lr - 0.80))
idx_knn = np.argmin(np.abs(tpr_knn - 0.80))

print("Logistic Regression")
print("TPR:", tpr_lr[idx_lr])
print("FPR:", fpr_lr[idx_lr])

print("\nKNN")
print("TPR:", tpr_knn[idx_knn])
print("FPR:", fpr_knn[idx_knn])


# KNN has the lower FPR (0.04) compared to Logistic Regression (0.55)
# when the TPR is close to 0.80. This means that if I wanted to catch
# about 80% of the positive cases, KNN would make fewer false alarms,
# so it would be the better model for this situation.

#************************************************************************** 

#*****************ROC Question 3****************
print('***************ROC Question 3***************')

#find the threshold that achieves the highest F1 score on the test set.
best_f1 = 0
best_threshold = 0
best_tpr = 0
best_fpr = 0

for i, threshold in enumerate(thresh_lr):

    predictions = (y_probs_lr >= threshold).astype(int)

    score = f1_score(y_test, predictions)

    if score > best_f1:
        best_f1 = score
        best_threshold = threshold
        best_tpr = tpr_lr[i]
        best_fpr = fpr_lr[i]

print("Best Threshold:", best_threshold)
print("TPR:", best_tpr)
print("FPR:", best_fpr)
print("Best F1:", best_f1)

# The best threshold (about 0.28) is lower than the default threshold of 0.5.
# Using a lower threshold predicts more positive cases, which increases the TPR,
# but it also increases the FPR. In a real application, I would choose a lower
# threshold when it is more important to detect as many positive cases as possible.

#*************************************************************************************
#*****GridSearchCV****************************
#***************GridSearch Question 1*********
print('***************GridSearch Question 1***************')

pipe_lr = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(max_iter=1000))
])

param_grid = {
    "lr__C":[0.001,0.01,0.1,1.0,10.0,100.0]
}

grid_lr = GridSearchCV(
    pipe_lr,
    param_grid,
    cv=5,
    scoring="roc_auc"
)

grid_lr.fit(X_train,y_train)

best_lr_pipe = grid_lr.best_estimator_

test_auc = roc_auc_score(
    y_test,
    best_lr_pipe.predict_proba(X_test)[:,1]
)

print("Best C:",grid_lr.best_params_["lr__C"])
print("Best CV AUC:",grid_lr.best_score_)
print("Test AUC:",test_auc)

# The grid search did not choose the default C value of 1.0. It selected
# C = 100.0 as the best value. The test AUC is about 0.7057, which is almost
# the same as the default logistic regression AUC (0.7060), so changing C
# did not make a noticeable difference on the test data.

# ***********************************************************************
#*******GridSearch Question 2*******************
print('***************GridSearch Question 2***************')


pipe_tree = Pipeline([
    ("scaler",StandardScaler()),
    ("tree",DecisionTreeClassifier(random_state=42))
])

tree_grid = {
    "tree__max_depth":[2,3,5,8,None]
}

grid_tree = GridSearchCV(
    pipe_tree,
    tree_grid,
    cv=5,
    scoring="roc_auc"
)

grid_tree.fit(X_train,y_train)

best_tree = grid_tree.best_estimator_

tree_auc = roc_auc_score(
    y_test,
    best_tree.predict_proba(X_test)[:,1]
)

print("Best max_depth:",grid_tree.best_params_["tree__max_depth"])
print("Best CV AUC:",grid_tree.best_score_)
print("Test AUC:",tree_auc)

# The best decision tree used a max_depth of 5. Its test AUC (0.9354)
# is much higher than the logistic regression test AUC (0.7057), so I
# would choose the decision tree for further development.
# I would also think about how simple
# the model is, how fast it runs, and whether it might overfit the data.

#************************************************************************
#********GridSearch Question 3*******************
print('***************GridSearch Question 3***************')
results = grid_lr.cv_results_

rows = []

for mean,std,param in zip(
    results["mean_test_score"],
    results["std_test_score"],
    results["param_lr__C"]
):
    rows.append((mean,std,param))

rows.sort(reverse=True)

for mean,std,param in rows:
    print(f"C={param}, Mean={mean:.4f}, Std={std:.4f}")

# C = 100.0 and C = 10.0 have almost the same mean AUC (0.7727 and 0.7726)
# and the same standard deviation (0.0057). If I had to choose between them,
# I would pick C = 100.0 because it has the slightly higher mean AUC. If two
# models had the same mean score but different standard deviations, I would
# choose the one with the smaller standard deviation because its performance
# is more consistent across the cross-validation folds.

#*************************************************************************
#******joblib***************************    
#****************  joblib Question 1*****
print('***************joblib Question 1***************')


joblib.dump(best_lr_pipe,"models/warmup_model.pkl")

loaded_clf = joblib.load("models/warmup_model.pkl")

original_preds = best_lr_pipe.predict(X_test)
loaded_preds = loaded_clf.predict(X_test)

assert (original_preds == loaded_preds).all(), "Predictions do not match!"

print("Predictions match. Model saved and loaded successfully.")

# If only the logistic regression model were saved,
# the scaler would be missing.
# The model would receive unscaled data and predictions could become inaccurate.

# ********joblib Question 2***************
print('***************joblib Question 2***************')    
# --- Simulated prediction script ---

loaded_model = joblib.load("models/warmup_model.pkl")

new_samples = np.array([
    [2.5,1.2,-0.3,0.8,1.0,-0.5,0.2,0.9,-1.1,0.4],
    [-1.0,0.5,0.9,-0.7,-0.2,1.3,-0.8,0.1,0.5,-0.3],
    [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
])

predictions = loaded_model.predict(new_samples)
probabilities = loaded_model.predict_proba(new_samples)

for i in range(len(new_samples)):
    print(f"\nSample {i+1}")
    print("Predicted class:", predictions[i])
    print("Probability:", probabilities[i])

# The all-zeros row is predicted as class 1 with a probability of about 0.65.
# This means the model thinks it is more likely to belong to the positive class.
# Even though all the values are zero, the prediction depends on what the model
# learned from the training data after the features were scaled.
