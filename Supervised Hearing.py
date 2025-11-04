# hearing_risk_prediction.py
# Author: Kiana Lang
# Course: CS492 - Machine Learning
# Date: September 14, 2025
# Description: Supervised machine learning model to predict hearing discomfort
#              using the 2025 Hearing Wellness Survey dataset.

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# -----------------------------
# Step 1: Load the dataset
# -----------------------------
DATA_PATH = "Hearing well-being Survey Report.csv"
dataset = pd.read_csv(DATA_PATH)

# -----------------------------
# Step 2: Clean the dataset
# -----------------------------
# Drop rows with missing target values
TARGET_COLUMN = 'Ear_Discomfort_After_Use'
dataset.dropna(subset=[TARGET_COLUMN], inplace=True)

# Drop columns with high cardinality or irrelevant text
columns_to_drop = ['Perceived_Hearing_Meaning', 'Desired_App_Features']
dataset.drop(columns=columns_to_drop, inplace=True)

# -----------------------------
# Step 3: Encode categorical variables
# -----------------------------
# Encode target variable
label_target = LabelEncoder()
dataset[TARGET_COLUMN] = label_target.fit_transform(dataset[TARGET_COLUMN])

# Encode all other object-type columns
for column in dataset.select_dtypes(include='object').columns:
    encoder = LabelEncoder()
    dataset[column] = encoder.fit_transform(dataset[column].astype(str))

# -----------------------------
# Step 4: Define features and target
# -----------------------------
features = dataset.drop(columns=[TARGET_COLUMN])
target = dataset[TARGET_COLUMN]

# -----------------------------
# Step 5: Scale the features
# -----------------------------
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# -----------------------------
# Step 6: Split the dataset
# -----------------------------
# Use 80% for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    features_scaled, target, test_size=0.2, random_state=42
)

# -----------------------------
# Step 7: Train the model
# -----------------------------
# Use Random Forest Classifier for robustness and interpretability
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# -----------------------------
# Step 8: Make predictions
# -----------------------------
predictions = model.predict(X_test)

# -----------------------------
# Step 9: Evaluate the model
# -----------------------------
print("\nClassification Report:")
print(classification_report(y_test, predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

# -----------------------------
# Step 10: Visualize feature importance
# -----------------------------
importance_series = pd.Series(model.feature_importances_, index=features.columns)
plt.figure(figsize=(10, 6))
importance_series.sort_values(ascending=False).plot(kind='bar')
plt.title('Feature Importances for Hearing Risk Prediction')
plt.ylabel('Importance')
plt.xlabel('Features')
plt.tight_layout()
plt.savefig('feature_importance_pep8.png')
plt.show()
print("Feature importance plot saved as 'feature_importance_pep8.png'.")
