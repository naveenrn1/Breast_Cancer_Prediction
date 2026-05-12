import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/breast_cancer.csv")
print(df.head())

print("data shape:")
print(df.shape)

print("columns:")
print(df.columns)

print("Data set info:")
print(df.info())

print("missing values:")
print(df.isnull().sum())

# we removed them because they are not usefull
if 'id' in df.columns:
    df.drop('id', axis=1, inplace=True)
if 'Unnamed:32' in df.columns:
    df.drop('Unnamed:32', axis=1, inplace=True)

print(df.head())

print(df['diagnosis'].value_counts())
sns.countplot(x='diagnosis', data=df)
plt.title("diagnosis count")
plt.show()

# convert diagonsis to numeric
df['diagnosis'] = df['diagnosis'].map({'M':1, 'B':0})
corr = df.corr()
# plot heatmap
plt.figure(figsize=(20,15))
sns.heatmap(corr, annot=False, cmap='coolwarm')
plt.title("correlation heatmap")
plt.show()
df.hist(figsize=(20,20))
plt.show()

plt.figure(figsize=(15,6))
sns.boxplot(data=df)
plt.xticks(rotation=90)
plt.title('Feature outliers')
plt.show()
print(df.describe())
df.to_csv("data/cleaned_breast_cancer.csv", index=False)
print("cleaned dataset saved")