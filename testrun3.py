# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report


# Load the dataset
data = pd.read_csv('data.csv')

# Display basic information about the dataset
print("Dataset shape:", data.shape)
print("\nFirst 5 rows of the dataset:")
print(data.head())

# Check for missing values
print("\nMissing values in each column:")
print(data.isnull().sum())

# Data cleaning and preprocessing
# 1. Convert string representations of numbers to actual numerical values

# Handle Purchase_Amount (remove $ and convert to float)
data['Purchase_Amount'] = data['Purchase_Amount'].str.replace('$', '').astype(float)

# Convert Age to integer
data['Age'] = pd.to_numeric(data['Age'], errors='coerce')

# Convert Time_Spent_on_Product_Research(hours) to float
data['Time_Spent_on_Product_Research(hours)'] = pd.to_numeric(data['Time_Spent_on_Product_Research(hours)'], errors='coerce')

# Convert boolean columns from string to actual boolean
boolean_cols = ['Discount_Used', 'Customer_Loyalty_Program_Member']
for col in boolean_cols:
    data[col] = data[col].map({'TRUE': 1, 'FALSE': 0})

# Check data types after conversion
print("\nData types after conversion:")
print(data.dtypes)

# Fill missing values
# For numerical columns, fill with mean
numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
for col in numerical_cols:
    if data[col].isnull().sum() > 0:
        data[col].fillna(data[col].mean(), inplace=True)

# For categorical columns, fill with mode (most frequent value)
categorical_cols = data.select_dtypes(include=['object']).columns
for col in categorical_cols:
    if data[col].isnull().sum() > 0:
        data[col].fillna(data[col].mode()[0], inplace=True)

# Check if all missing values are handled
print("\nRemaining missing values after imputation:")
print(data.isnull().sum())

# Encode categorical variables
label_encoders = {}
for col in categorical_cols:
    if col != 'Customer_ID':  # Skip Customer_ID as it's just an identifier
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

# Define features (X) and target variable (y)
# Using 'Purchase_Intent' as the target variable
X = data.drop(['Customer_ID', 'Purchase_Intent'], axis=1)
y = data['Purchase_Intent']

# Print unique values in the target variable
print("\nUnique values in Purchase_Intent:")
print(data['Purchase_Intent'].unique())
print("\nDistribution of Purchase_Intent:")
print(data['Purchase_Intent'].value_counts())

# Scale numerical features for better model performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# Split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on test set
y_pred = model.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature importance analysis
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nFeature Importance (Top 10):")
print(feature_importance.head(10))

# Visualize feature importance
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10))
plt.title('Top 10 Most Important Features for Purchase Intent Prediction')
plt.tight_layout()
plt.savefig('feature_importance.png')
print("Feature importance plot saved as 'feature_importance.png'")

# Create example test cases based on actual data
# Extract a few rows from the original dataset to use as test cases
test_samples = data.sample(4)
print("\nExample test cases (from actual data):")
print(test_samples[['Age', 'Gender', 'Income_Level', 'Purchase_Category', 'Purchase_Amount']])

# Prepare test data (excluding Customer_ID and Purchase_Intent)
test_X = test_samples.drop(['Customer_ID', 'Purchase_Intent'], axis=1)

# Scale the test data
test_X_scaled = scaler.transform(test_X)

# Make predictions on test cases
test_predictions = model.predict(test_X_scaled)

# Get the original Purchase_Intent values for comparison
actual_intents = test_samples['Purchase_Intent'].values

# Display results
print("\nPredictions for Example Test Cases:")
for i, (pred, actual) in enumerate(zip(test_predictions, actual_intents)):
    print(f"Test Case {i+1}:")
    print(f"  Predicted Purchase Intent: {pred}")
    print(f"  Actual Purchase Intent: {actual}")
    print(f"  Correct: {'Yes' if pred == actual else 'No'}")

# Visualize the distribution of the target variable
plt.figure(figsize=(10, 6))
sns.countplot(x=y)
plt.title('Distribution of Purchase Intent')
plt.savefig('purchase_intent_distribution.png')
print("Purchase intent distribution plot saved as 'purchase_intent_distribution.png'")

# Correlation matrix for numerical features
numerical_data = data.select_dtypes(include=['float64', 'int64'])
plt.figure(figsize=(14, 10))
correlation_matrix = numerical_data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix of Numerical Features')
plt.tight_layout()
plt.savefig('correlation_matrix.png')
print("Correlation matrix plot saved as 'correlation_matrix.png'")

print("\nAnalysis complete!")
print(data.head())