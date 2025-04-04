import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

def encode_data(data, categorical_cols):
    """Encode categorical columns and return encoded data with encoders"""
    encoded_data = data.copy()
    label_encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        encoded_data[col] = le.fit_transform(data[col].astype(str))
        label_encoders[col] = le
    
    return encoded_data, label_encoders

def preprocess_data(data):
    """Clean and preprocess the data"""
    # Remove dollar signs and convert to float
    data['Purchase_Amount'] = data['Purchase_Amount'].str.replace('$', '').str.replace(',', '').astype(float)
    
    # Convert date to datetime
    data['Time_of_Purchase'] = pd.to_datetime(data['Time_of_Purchase'])
    
    return data

def train_model(X, y, test_size=0.2, random_state=42):
    """Train the model and return model with metrics"""
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Initialize and train model
    model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    model.fit(X_train, y_train)
    
    # Get predictions
    y_pred = model.predict(X_test)
    
    # Print classification report
    print("\nModel Performance:")
    print(classification_report(y_test, y_pred))
    
    return model, (X_test, y_test, y_pred)

def save_model(model, label_encoders, feature_columns, path='model/'):
    """Save the model and associated transformers"""
    import os
    os.makedirs(path, exist_ok=True)
    
    joblib.dump(model, f'{path}model.pkl')
    joblib.dump(label_encoders, f'{path}label_encoders.pkl')
    joblib.dump(feature_columns, f'{path}feature_columns.pkl')

if __name__ == "__main__":
    # Load and preprocess data
    data = pd.read_csv('data.csv')
    data = preprocess_data(data)
    
    # Define features and target
    feature_columns = [
        'Age', 'Gender', 'Income_Level', 'Purchase_Amount', 
        'Purchase_Category', 'Device_Used_for_Shopping',
        'Time_Spent_on_Product_Research(hours)', 'Discount_Used', 
        'Customer_Loyalty_Program_Member'
    ]
    
    target_column = 'Purchase_Intent'
    
    # Prepare features and target
    X = data[feature_columns]
    y = data[target_column]
    
    # Identify categorical columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    
    # Encode categorical data
    X_encoded, label_encoders = encode_data(X, categorical_cols)
    
    # Train model
    model, evaluation_data = train_model(X_encoded, y)
    
    # Save model and transformers
    save_model(model, label_encoders, feature_columns)
    print("\nModel and transformers saved successfully!")
