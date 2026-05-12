import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# load data set 
df = pd.read_csv("data/breast_cancer.csv")
print(df.head())

# drop unessry columns
if 'id' in df.columns:
    df.drop('id', axis=1, inplace=True)

if 'Unnamed: 32' in df.columns:
    df.drop('Unnamed: 32', axis=1, inplace=True)

# encloded diagnosis column
encoder = LabelEncoder()
df['diagnosis'] = encoder.fit_transform(df['diagnosis'])
print(df.head())

# Feature and target
selected_features = [
    'radius_mean',
    'texture_mean',
    'perimeter_mean',
    'area_mean',
    'smoothness_mean',
    'compactness_mean',
    'concavity_mean',
    'concave points_mean',
    'symmetry_mean',
    'fractal_dimension_mean'
]
X = df[selected_features]
y = df['diagnosis']
print("features shape", X.shape)
print("Target shape", y.shape)

# train-test split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)
print("traning data", X_train.shape)
print("testing data", X_test.shape)


# train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predection
y_pred = model.predict(X_test)

# accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"accuracy:{accuracy*100:.2f}%")


# logistic regression
lr_model = LogisticRegression(max_iter=5000)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_pred)
print("logistic regression accuracy:", lr_accuracy*100)

# decission tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_pred)
print("Decssion tree Accuracy:", dt_accuracy*100)

# Random forest

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
print("random forest:", rf_accuracy*100)

# support vector machine  (svm)

svm_model = SVC()
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_pred)
print("svm accuracy:", svm_accuracy*100)

print("=================")
print("model accuracy comparision")
print("===================")

print(f"Logistic regression:{lr_accuracy*100:.2f}%")
print(f"dession tree:{dt_accuracy*100:.2f}%")
print(f"random forest:{rf_accuracy*100:.2f}%")
print(f"svm:{svm_accuracy*100:.2f}%")

print("confusion matrix")
print(confusion_matrix(y_test, rf_pred))

print("\nClassification Report:")

print(classification_report(y_test, rf_pred))


# save model
pickle.dump(model, open("model/model.pkl", "wb"))
print("model saved sucessfully")

print(df.columns)