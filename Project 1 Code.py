import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

## QUESTION 1 ##

# Read the data

data = pd.read_csv('data/project_1_data.csv')

df = pd.DataFrame(data)

## QUESTION 2 ##

# Statistical analysis

print('Shape of the data:')
print('\n')
print('Rows, Columns')
print(df.shape)
print('\n')

print('First five rows of data:')
print('\n')
print(df.head())
print('\n')

print('Statistical Summary:')
print('\n')
print(df.describe())
print('\n')

# Visualization of the data

for col in ['X', 'Y', 'Z']:
    plt.figure(figsize=(6,4))
    plt.hist(df[col], bins=25, color='lightblue', edgecolor='black')
    plt.title(f'{col} Distribution')
    plt.xlabel(f'{col} Value')
    plt.ylabel('Frequency')
    plt.grid(alpha=0.3)
    plt.show()

plt.figure(figsize=(6,4))
plt.hist(df['Step'], bins=13, color='lightblue', edgecolor='black')
plt.title('Step Distribution')
plt.xlabel('Step Value')
plt.ylabel('Frequency')
plt.grid(alpha=0.3)
plt.show()

## QUESTION 3 ##

# Correlation matrix creation and plotting

corr = df.corr(method = 'pearson')

print('Correlation Matrix:')
print('\n')
print(corr)
print('\n')

plt.figure()
sns.heatmap(corr, annot = True)
plt.title('Correlation Matrix')
plt.show()

## QUESTION 4 ##

# Define variables for ML models

X = df[['X', 'Y', 'Z']]
y = df['Step']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)

# Model 1: Random Forest

mdl1 = RandomForestClassifier(random_state = 42)

mdl1_params = {
    'n_estimators': [100, 200, 500],       
    'max_depth': [None, 10, 20, 30],       
    'min_samples_split': [2, 5, 10],           
    'max_features': ['sqrt', 'log2'] 
    }

mdl1_grid = GridSearchCV(mdl1, mdl1_params, cv = 5, n_jobs = -1, scoring='accuracy')
mdl1_grid.fit(X_train, y_train)

print('The Best Parameters for the Random Forest Model Are:')
for param, value in mdl1_grid.best_params_.items():
    print(f"  {param}: {value}")
print("Training Accuracy:", accuracy_score(y_train, mdl1_grid.best_estimator_.predict(X_train)))
print("Testing Accuracy:", accuracy_score(y_test, mdl1_grid.best_estimator_.predict(X_test)))
print('\n')

# Model 2: Logistic Regression

mdl2 = LogisticRegression(max_iter = 5000)
mdl2_params = {
    'C': [0.01, 0.1, 1, 10], 
    'penalty': ['l2'], 
    'solver': ['lbfgs', 'liblinear']
    }


mdl2_grid = GridSearchCV(mdl2, mdl2_params, cv = 5, n_jobs = -1, scoring='accuracy')
mdl2_grid.fit(X_train, y_train)

print('The Best Parameters for the Logistic Regression Model Are:')
for param, value in mdl2_grid.best_params_.items():
    print(f"  {param}: {value}")
print("Training Accuracy:", accuracy_score(y_train, mdl2_grid.best_estimator_.predict(X_train)))
print("Testing Accuracy:", accuracy_score(y_test, mdl2_grid.best_estimator_.predict(X_test)))
print('\n')

# Model 3: SVM

mdl3 = SVC()
mdl3_params = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto']
}

mdl3_grid = GridSearchCV(mdl3, mdl3_params, cv = 5, n_jobs = -1, scoring='accuracy')
mdl3_grid.fit(X_train, y_train)

print('The Best Parameters for the SVM Model Are:')
for param, value in mdl2_grid.best_params_.items():
    print(f"  {param}: {value}")
print("Training Accuracy:", accuracy_score(y_train, mdl3_grid.best_estimator_.predict(X_train)))
print("Testing Accuracy:", accuracy_score(y_test, mdl3_grid.best_estimator_.predict(X_test)))
print('\n')

# Model 4: Random Forest with Random Search

mdl4 = RandomForestClassifier(random_state=42)
mdl4_params = {
    'n_estimators': [50, 100, 200, 300, 500],
    'max_depth': [None, 10, 20, 30, 50],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 4, 6],
    'max_features': ['sqrt', 'log2', None]
}

mdl4_random = RandomizedSearchCV(mdl4, mdl4_params, n_iter = 30, cv = 5, n_jobs = -1, scoring='accuracy')
mdl4_random.fit(X_train, y_train)

print('The Best Parameters for the Random Forest Model with RandomSearchCV Are:')
for param, value in mdl4_random.best_params_.items():
    print(f"  {param}: {value}")

print("Training Accuracy:", accuracy_score(y_train, mdl4_random.best_estimator_.predict(X_train)))
print("Testing Accuracy:", accuracy_score(y_test, mdl4_random.best_estimator_.predict(X_test)))
print('\n')


