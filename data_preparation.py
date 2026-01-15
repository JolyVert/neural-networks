import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

CSV_PATH = 'data/Titanic-Dataset.csv'
OUTPUT_DIR = 'data'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print("Loading data...")
try:
    df = pd.read_csv(CSV_PATH)
except FileNotFoundError:
    print(f"ERROR: File {CSV_PATH} not found. Please download it and place it in the script folder.")
    exit()

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
target = 'Survived'

X = df[features].copy()
y = df[target].copy()

X['Sex'] = X['Sex'].map({'male': 0, 'female': 1})

X['Age'] = X['Age'].fillna(X['Age'].median())
X['Embarked'] = X['Embarked'].fillna(X['Embarked'].mode()[0])
X['Fare'] = X['Fare'].fillna(X['Fare'].median())

X = pd.get_dummies(X, columns=['Embarked'], drop_first=True)

scaler = StandardScaler()
X_scaled_array = scaler.fit_transform(X)
X_processed = pd.DataFrame(X_scaled_array, columns=X.columns)

print(f"Saving files to folder '{OUTPUT_DIR}'...")

joblib.dump(X_processed, os.path.join(OUTPUT_DIR, 'X_train_processed.pkl'))
joblib.dump(y.values, os.path.join(OUTPUT_DIR, 'y_train_processed.pkl'))
joblib.dump(scaler, os.path.join(OUTPUT_DIR, 'titanic_scaler.pkl'))

print("Done! Data has been prepared.")
print(f"Number of features (INPUT_SIZE): {X_processed.shape[1]}")
print(f"Number of samples: {len(X_processed)}")
