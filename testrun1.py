# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
data = pd.read_csv('data.csv')

# Display first few rows to understand the data structure
print("Dataset Preview:")
print(data.head())

# Get basic information about the dataset
print("\nDataset Information:")
print(data.info())

# Check for missing values in each column
print("\nMissing Values Count:")
print(data.isnull().sum())

# Check for duplicate rows
print("\nNumber of Duplicate Rows:", data.duplicated().sum())

# Display basic statistics for numerical columns
print("\nBasic Statistics for Numerical Columns:")
print(data.describe())

# Data Preprocessing
print("\nStarting Data Preprocessing...")

# Handle missing values
# For numerical columns, fill with mean
numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
for col in numerical_cols:
    if data[col].isnull().sum() > 0:
        data[col].fillna(data[col].mean(), inplace=True)

# For categorical columns, fill with most frequent value
categorical_cols = data.select_dtypes(include=['object']).columns
for col in categorical_cols:
    if data[col].isnull().sum() > 0:
        data[col].fillna(data[col].mode()[0], inplace=True)

# Handle specific columns with missing values mentioned in your data
data['Social_Media_Influence'].fillna(data['Social_Media_Influence'].mean(), inplace=True)
data['Engagement_with_Ads'].fillna(data['Engagement_with_Ads'].mean(), inplace=True)

# Check if 'Time_to_Decision' has missing values and handle them
if 'Time_to_Decision' in data.columns and data['Time_to_Decision'].isnull().sum() > 0:
    data['Time_to_Decision'].fillna(data['Time_to_Decision'].mean(), inplace=True)

# Verify all missing values are handled
print("\nRemaining Missing Values After Imputation:")
print(data.isnull().sum())

# Encode categorical variables
print("\nEncoding Categorical Variables...")
encoders = {}  # Store encoders for later use in predictions

for column in categorical_cols:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    encoders[column] = le

# Feature Selection
# Select relevant features for predicting Purchase_Intent
# Exclude the target variable from features
target_column = 'Purchase_Intent'
feature_columns = [col for col in data.columns if col != target_column]

# Create feature matrix and target vector
X = data[feature_columns]
y = data[target_column]

# Encode target if it's categorical
if y.dtype == 'object':
    y = encoders.get(target_column, LabelEncoder()).fit_transform(y)

# Standardize numerical features (optional but often helpful)
scaler = StandardScaler()
X_scaled = X.copy()
for col in numerical_cols:
    if col in X_scaled.columns:
        X_scaled[col] = scaler.fit_transform(X_scaled[[col]])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
print(f"\nSplit data into training set ({X_train.shape[0]} samples) and test set ({X_test.shape[0]} samples)")

# Train a Random Forest model
print("\nTraining Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on test set
y_pred = model.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.4f}")

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))

# Visualize feature importance
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance.head(15))
plt.title('Feature Importance for Purchase Intent Prediction')
plt.tight_layout()
plt.savefig('feature_importance.png')
print("\nFeature importance plot saved as 'feature_importance.png'")

# Example test data - create sample test cases
print("\nMaking predictions on sample test cases...")
# Create 4 sample customers with different profiles
test_data = []

# For demonstration, we'll create sample data
# You would replace these values with actual test data
# First, get the column order from X
columns_order = X.columns.tolist()

# Create sample test data (adjust values based on your actual data)
# Sample 1: Young, high income, digital shopper
# Sample 2: Middle-aged, medium income, traditional shopper
# Sample 3: Senior, low income, mixed shopping preference
# Sample 4: Young professional, high income, brand loyal

# For simplicity, I'll use random values within reasonable ranges
# In a real scenario, you'd create more meaningful test cases
np.random.seed(42)
test_data = []
for i in range(4):
    sample = {}
    for col in columns_order:
        if col in numerical_cols:
            # Use random value within the range of that column
            min_val = data[col].min()
            max_val = data[col].max()
            sample[col] = np.random.uniform(min_val, max_val)
        else:
            # For categorical columns, use a random existing category
            unique_values = data[col].unique()
            sample[col] = np.random.choice(unique_values)
    test_data.append(sample)

# Convert test data to DataFrame
test_df = pd.DataFrame(test_data)

# Make predictions
predictions = model.predict(test_df)

# If the target was encoded, convert predictions back to original labels
if target_column in encoders:
    predicted_labels = encoders[target_column].inverse_transform(predictions)
else:
    predicted_labels = predictions

# Display results
print("\nPredictions for Sample Test Cases:")
for i, pred in enumerate(predicted_labels):
    print(f"Test Case {i+1}: Predicted Purchase Intent -> {pred}")
