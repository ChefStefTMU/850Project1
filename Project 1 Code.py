import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

## QUESTION 1 ##

df = pd.read_csv('data/project_1_data.csv')
df = df.dropna().reset_index(drop=True)
df['Step'] = df['Step'].astype(int)

## QUESTION 2 ##

print('Shape of the Data:')
print(df.shape, '\n')

print('Statistical Summary:')
print(df.describe(), '\n')

print('First Five Rows of Data:')
print(df.head(), '\n')

# VISUALIZE DISTRIBUTIONS
for data_columns in ['X', 'Y', 'Z']:
    plt.hist(df[data_columns], bins=25, color='lightblue', edgecolor='black')
    plt.title(f'{data_columns} Distribution')
    plt.xlabel(f'{data_columns} Value')
    plt.ylabel('Frequency')
    plt.show()

plt.hist(df['Step'], bins=len(df['Step'].unique()), color='lightblue', edgecolor='black')
plt.title('Step Distribution')
plt.xlabel('Step Value')
plt.ylabel('Frequency')
plt.show()

## QUESTION 3 ##

# CORRELATION MATRIX
corr = df.corr(method='pearson')
print('Correlation Matrix:\n', corr, '\n')

sns.heatmap(corr, annot=True)
plt.title('Pearson Correlation Heatmap')
plt.show()

## QUESTION 4 ##

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, log_loss, confusion_matrix

# DEFINE VARIABLES
X = df[['X', 'Y', 'Z']]
y = df['Step']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# RANDOM FOREST MODEL (MODEL 1)
mdl1 = RandomForestClassifier(random_state=42)
mdl1_params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}
mdl1_search = GridSearchCV(mdl1, mdl1_params, scoring='accuracy', n_jobs=-1, cv=5, verbose=0)
mdl1_search.fit(X_train, y_train)
best_mdl1 = mdl1_search.best_estimator_
y_pred1 = best_mdl1.predict(X_test)

print('\nBest Hyperparameters for Random Forest:')
for param, value in mdl1_search.best_params_.items():
    print(f'{param}: {value}')

print('\nTest Accuracy (Random Forest):', accuracy_score(y_test, y_pred1))
print('\nClassification Report (Random Forest):\n', classification_report(y_test, y_pred1))
y_pred_proba1 = best_mdl1.predict_proba(X_test)
print('\nCross-Entropy Loss (Random Forest):', log_loss(y_test, y_pred_proba1))
cv_scores1 = cross_val_score(best_mdl1, X, y, cv=5, scoring='accuracy')
print('\nCross-validation scores (Random Forest):', cv_scores1)
print('Mean CV accuracy:', cv_scores1.mean())
print('Standard deviation:', cv_scores1.std())

# LOGISTIC REGRESSION (MODEL 2)
pipe_logreg = Pipeline([
    ('scaler', StandardScaler()),
    ('log_reg', LogisticRegression(max_iter=1000, random_state=42))
])
mdl2_params = {
    'log_reg__C': [0.01, 0.1, 1, 10],
    'log_reg__penalty': ['l2'],
    'log_reg__solver': ['lbfgs', 'saga']
}
mdl2_search = GridSearchCV(pipe_logreg, mdl2_params, scoring='accuracy', n_jobs=-1, cv=5, verbose=0)
mdl2_search.fit(X_train, y_train)
best_mdl2 = mdl2_search.best_estimator_
y_pred2 = best_mdl2.predict(X_test)

print('\nBest Hyperparameters for Logistic Regression:')
for param, value in mdl2_search.best_params_.items():
    print(f'{param}: {value}')

print('\nTest Accuracy (Logistic Regression):', accuracy_score(y_test, y_pred2))
print('\nClassification Report (Logistic Regression):\n', classification_report(y_test, y_pred2))
y_pred_proba2 = best_mdl2.predict_proba(X_test)
print('\nCross-Entropy Loss (Logistic Regression):', log_loss(y_test, y_pred_proba2))
cv_scores2 = cross_val_score(best_mdl2, X, y, cv=5, scoring='accuracy')
print('\nCross-validation scores (Logistic Regression):', cv_scores2)
print('Mean CV accuracy:', cv_scores2.mean())
print('Standard deviation:', cv_scores2.std())

# SVM MODEL (MODEL 3)
pipe_svm = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(probability=True, random_state=42))
])
svm_params = {
    'svm__C': [0.1, 1, 10],
    'svm__kernel': ['linear', 'rbf'],
    'svm__gamma': ['scale', 'auto']
}
svm_search = GridSearchCV(pipe_svm, svm_params, scoring='accuracy', n_jobs=-1, cv=5, verbose=0)
svm_search.fit(X_train, y_train)
best_svm = svm_search.best_estimator_
y_pred_svm = best_svm.predict(X_test)

print('\nBest Hyperparameters for SVM:')
for param, value in svm_search.best_params_.items():
    print(f'{param}: {value}')

print('\nTest Accuracy (SVM):', accuracy_score(y_test, y_pred_svm))
print('\nClassification Report (SVM):\n', classification_report(y_test, y_pred_svm))
y_pred_proba_svm = best_svm.predict_proba(X_test)
print('\nCross-Entropy Loss (SVM):', log_loss(y_test, y_pred_proba_svm))
cv_scores_svm = cross_val_score(best_svm, X, y, cv=5, scoring='accuracy')
print('\nCross-validation scores (SVM):', cv_scores_svm)
print('Mean CV accuracy:', cv_scores_svm.mean())
print('Standard deviation:', cv_scores_svm.std())

## QUESTION 5 ##

# Confusion Matrices
cm1 = confusion_matrix(y_test, y_pred1)
sns.heatmap(cm1, annot=True, fmt='d', cmap='Oranges')
plt.title('Confusion Matrix - Random Forest')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

cm2 = confusion_matrix(y_test, y_pred2)
sns.heatmap(cm2, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Logistic Regression')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

cm3 = confusion_matrix(y_test, y_pred_svm)
sns.heatmap(cm3, annot=True, fmt='d', cmap='Greens')
plt.title('Confusion Matrix - SVM')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()