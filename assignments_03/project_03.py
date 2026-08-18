#Part 2: Mini-Project -- Spam or Ham? A Classifier Shootout
#Jarirtech

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from io import BytesIO
import os


from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
from sklearn.inspection import DecisionBoundaryDisplay

warnings.filterwarnings("ignore", category=RuntimeWarning)
os.makedirs("outputs", exist_ok=True)




#--------Task 1: Load and Explore------------------
print('#--------Task 1: Load and Explore------------------')

#Loading the Dataset
COLUMN_NAMES = [
    "word_freq_make",        # 0   percent of words that are "make"
    "word_freq_address",     # 1
    "word_freq_all",         # 2
    "word_freq_3d",          # 3   almost never appears
    "word_freq_our",         # 4
    "word_freq_over",        # 5
    "word_freq_remove",      # 6   common in "remove me from this list"
    "word_freq_internet",    # 7
    "word_freq_order",       # 8
    "word_freq_mail",        # 9
    "word_freq_receive",     # 10
    "word_freq_will",        # 11
    "word_freq_people",      # 12
    "word_freq_report",      # 13
    "word_freq_addresses",   # 14
    "word_freq_free",        # 15  classic spam word
    "word_freq_business",    # 16
    "word_freq_email",       # 17
    "word_freq_you",         # 18
    "word_freq_credit",      # 19
    "word_freq_your",        # 20  often high in spam
    "word_freq_font",        # 21  HTML emails
    "word_freq_000",         # 22  "win $ x,000" style offers
    "word_freq_money",       # 23  money related
    "word_freq_hp",          # 24  HP specific
    "word_freq_hpl",         # 25
    "word_freq_george",      # 26  specific HP person
    "word_freq_650",         # 27  area code
    "word_freq_lab",         # 28
    "word_freq_labs",        # 29
    "word_freq_telnet",      # 30
    "word_freq_857",         # 31
    "word_freq_data",        # 32
    "word_freq_415",         # 33
    "word_freq_85",          # 34
    "word_freq_technology",  # 35
    "word_freq_1999",        # 36
    "word_freq_parts",       # 37
    "word_freq_pm",          # 38
    "word_freq_direct",      # 39
    "word_freq_cs",          # 40
    "word_freq_meeting",     # 41
    "word_freq_original",    # 42
    "word_freq_project",     # 43
    "word_freq_re",          # 44  reply threads
    "word_freq_edu",         # 45
    "word_freq_table",       # 46
    "word_freq_conference",  # 47
    "char_freq_;",           # 48  frequency of ';'
    "char_freq_(",           # 49  frequency of '('
    "char_freq_[",           # 50  frequency of '['
    "char_freq_!",           # 51  exclamation marks (often big)
    "char_freq_$",           # 52  dollar sign (money related)
    "char_freq_#",           # 53  hash character
    "capital_run_length_average",  # 54  average length of capital letter runs
    "capital_run_length_longest",  # 55  longest capital run
    "capital_run_length_total",    # 56  total number of capital letters
    "spam_label"                    # 57  1 = spam, 0 = not spam
]




url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
response = requests.get(url)
response.raise_for_status()

df = pd.read_csv(BytesIO(response.content), header=None)
df.columns = COLUMN_NAMES
print(df.head())
print(df.shape)
print(df["spam_label"].value_counts())# 0    2788  + 1    1813 --> 4601 emails


# create a boxplot showing the distribution of that feature for spam emails versus ham emails


features = [
    "word_freq_free",
    "char_freq_!",
    "capital_run_length_total"
]

for feature in features:
    plt.figure(figsize=(6,4))
    df.boxplot(column=feature, by="spam_label")
    plt.title(feature)
    plt.suptitle("")
    plt.xlabel("0 = ham , 1 = spam")
    plt.savefig(f"outputs/{feature}_boxplot.png")
    plt.close()


# I notice that spam emails (1) usually have higher values for word_freq_free
# than ham emails (0). Ham emails are mostly near zero, while spam has more
# spread and many higher outliers.

# The difference between classes is noticeable, so it is more dramatic than subtle.
# The word "free" appears much more often in spam emails.

# The heavy skew toward zero means many emails do not contain this word at all,
# so the data is sparse with many zero values.
# the same for char_freq_!, and capital_run_length_total
#----------------------------------------------------------------------------------------


#--------Task 2: Prepare Your Data------------------
print('#--------Task 2: Prepare Your Data------------------')
X = df.drop("spam_label", axis=1)
y = df["spam_label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# only on train avoiding leakage
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# PCA
pca = PCA()
pca.fit(X_train_scaled)

cum_var = np.cumsum(pca.explained_variance_ratio_)

plt.figure(figsize=(8,5))
plt.plot(range(1, len(cum_var)+1), cum_var)
plt.axhline(0.90, linestyle="--")
plt.xlabel("Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Cumulative Explained Variance")
plt.savefig("outputs/pca_cumulative_explained_variance.png")

plt.close()
    
n = np.argmax(cum_var >= 0.90) + 1
print("Components for 90% variance:", n)

X_train_pca = pca.transform(X_train_scaled)[:, :n]
X_test_pca = pca.transform(X_test_scaled)[:, :n]

# Scaling features and fitting PCA on training only  avoiding data leakage.

#----------------------------------------------------------------------------------------

#--------Task 3: A Classifier Comparison------------------
print('#--------Task 3: A Classifier Comparison------------------')

# -----------------------------
# KNN Unscaled
# -----------------------------
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

knn_pred = knn.predict(X_test)

print("\nKNN Unscaled")
print("Accuracy:", accuracy_score(y_test, knn_pred))
print(classification_report(y_test, knn_pred))

# -----------------------------
# KNN Scaled
# -----------------------------
knn.fit(X_train_scaled, y_train)

knn_scaled_pred = knn.predict(X_test_scaled)

print("\nKNN Scaled")
print("Accuracy:", accuracy_score(y_test, knn_scaled_pred))
print(classification_report(y_test, knn_scaled_pred))

# -----------------------------
# KNN PCA
# -----------------------------
knn.fit(X_train_pca, y_train)

knn_pca_pred = knn.predict(X_test_pca)

print("\nKNN PCA")
print("Accuracy:", accuracy_score(y_test, knn_pca_pred))
print(classification_report(y_test, knn_pca_pred))

# -----------------------------
# Decision Tree Depth Comparison
# -----------------------------
print("\nDecision Tree Depth Comparison")

for depth in [3, 5, 10, None]:

    tree = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42
    )

    tree.fit(X_train, y_train)

    train_acc = tree.score(X_train, y_train)
    test_acc = tree.score(X_test, y_test)

    print(
        f"Depth={depth}, "
        f"Train={train_acc:.4f}, "
        f"Test={test_acc:.4f}"
    )

# Best Decision Tree
tree = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

tree.fit(X_train, y_train)

tree_pred = tree.predict(X_test)

print("\nDecision Tree")
print("Accuracy:", accuracy_score(y_test, tree_pred))
print(classification_report(y_test, tree_pred))

# Top 10 Decision Tree Features
tree_importance = pd.Series(
    tree.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nTop 10 Decision Tree Features")
print(tree_importance.head(10))

# As tree depth increases, training accuracy increases while the gap
# between training and test accuracy also grows, showing overfitting.
# Although the fully grown tree achieved the highest test accuracy,
# deeper trees are more likely to memorize the training data.

# -----------------------------
# Random Forest
# -----------------------------
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

print("\nRandom Forest")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print(classification_report(y_test, rf_pred))

# Top 10 Random Forest Features
importance = pd.Series(
    rf.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nTop 10 Random Forest Features")
print(importance.head(10))

plt.figure(figsize=(10,6))
importance.head(10).plot(kind="bar")
plt.title("Random Forest Feature Importances")
plt.tight_layout()
plt.savefig("outputs/feature_importances.png")
plt.close()

# Both models identify important spam indicators such as char_freq_$,
# char_freq_!, word_freq_remove, word_freq_free, and word_freq_hp.
# Random Forest also ranks several capital-letter features highly.
# The models generally agree on the most important predictors, but
# Random Forest produces more stable importance estimates because
# it averages many trees instead of relying on a single tree.

# -----------------------------
# Logistic Regression (Scaled)
# -----------------------------
lr = LogisticRegression(
    C=1.0,
    max_iter=1000,
    solver="liblinear"
)

lr.fit(X_train_scaled, y_train)

lr_pred = lr.predict(X_test_scaled)

print("\nLogistic Regression Scaled")
print("Accuracy:", accuracy_score(y_test, lr_pred))
print(classification_report(y_test, lr_pred))

# -----------------------------
# Logistic Regression + PCA
# -----------------------------
lr.fit(X_train_pca, y_train)

lr_pca_pred = lr.predict(X_test_pca)

print("\nLogistic Regression PCA")
print("Accuracy:", accuracy_score(y_test, lr_pca_pred))
print(classification_report(y_test, lr_pca_pred))

# -----------------------------
# Best Model Confusion Matrix
# -----------------------------
cm = confusion_matrix(y_test, rf_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

plt.title("Best Model Confusion Matrix")
plt.savefig("outputs/best_model_confusion_matrix.png")
plt.close()

# The Random Forest produces slightly more false negatives than false positives.
# False negatives allow spam messages into the inbox, which is generally
# a more costly error than incorrectly filtering a legitimate email.
# Therefore, improving recall for the spam class is especially important.
# -----------------------------
# Overall Comparison
# -----------------------------


# KNN:
# Scaling improves KNN substantially because distance calculations are
# affected by feature magnitude. PCA keeps accuracy close to the scaled
# model but does not improve it further.
# PCA reduces dimensionality, but scaled KNN performs slightly better.

# Logistic Regression:
# Scaling improves Logistic Regression because all features contribute
# on the same scale. PCA slightly lowers accuracy because compressing
# the data removes some useful information.

# Decision Tree vs Random Forest:
# Random Forest performs better than a single Decision Tree because it
# combines many trees, reducing overfitting and improving generalization.

# Spam Filter Discussion:
# For spam filtering, false negatives are usually worse than false positives.
# Allowing spam into the inbox is generally more harmful than incorrectly
# sending a legitimate email to the spam folder, so recall for the spam
# class is especially important.
#--------Task 4: Cross-Validation------------------
print('#--------Task 4: Cross-Validation------------------')

models = {
    "KNN Scaled": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(5))
    ]),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=1000,
            solver="liblinear"
        ))
    ])
}

for name, model in models.items():

    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=5
    )

    print(name)
    print("Mean:", scores.mean())
    print("Std :", scores.std())
    print()



# The most accurate model is Random Forest because it has the highest
# mean cross-validation score (0.9541).

# The most stable model is Logistic Regression because it has the lowest
# standard deviation across folds (0.0077), which means its performance
# was the most consistent.


# Yes, this mostly matches the single train/test split results because
# Random Forest was also one of the strongest models there.
# Cross-validation gives more confidence because it uses multiple folds
# instead of relying on one split.

#----------------------------------------------------------------------------------------


#--------Task 5: Building a Prediction Pipeline------------------
print('#--------Task 5: Building a Prediction Pipeline------------------')

# Best Tree Model
tree_pipe = Pipeline([
    ("model", RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])

tree_pipe.fit(X_train, y_train)

pipe_pred = tree_pipe.predict(X_test)

print("\nTree Pipeline")
print(classification_report(y_test, pipe_pred))

# Best Non Tree Model
non_tree_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=n)),
    ("model", LogisticRegression(
        max_iter=1000,
        solver="liblinear"
    ))
])

non_tree_pipe.fit(X_train, y_train)

pipe2_pred = non_tree_pipe.predict(X_test)

print("\nNon Tree Pipeline")
print(classification_report(y_test, pipe2_pred))

# comments:
# The tree pipeline uses only RandomForest because tree-based models do not require scaling or PCA.
# The non-tree pipeline includes scaling and PCA because Logistic Regression depends on standardized inputs.
# Pipelines bundle preprocessing + modeling into one object, preventing leakage and ensuring consistent steps during training and deployment.
# The two pipelines do not have the same structure: the tree-based pipeline contains only the model
# because trees do not require scaling or PCA. The non-tree pipeline includes scaling and PCA because
# Logistic Regression depends on standardized inputs and benefits from dimensionality reduction.
# Packaging preprocessing with the model ensures the exact same steps are applied during training,
# evaluation, and deployment, preventing data leakage and making the workflow reproducible.
