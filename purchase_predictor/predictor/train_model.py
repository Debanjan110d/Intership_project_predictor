import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os

def train_and_save_model():
    # Create models directory if it doesn't exist
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Load and prepare data
    data = pd.read_csv('../../data.csv')
    
    # Define features to use
    categorical_cols = ['Gender', 'Income_Level', 'Purchase_Category', 'Device_Used_for_Shopping']
    numeric_cols = ['Age', 'Purchase_Amount', 'Time_Spent_on_Product_Research(hours)']
    binary_cols = ['Discount_Used', 'Customer_Loyalty_Program_Member']
    
    # Prepare feature matrix
    feature_columns = categorical_cols + numeric_cols + binary_cols
    X = data[feature_columns].copy()
    y = data['Purchase_Intent']
    
    # Initialize label encoders for categorical variables
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
    
    # Convert boolean columns to int
    for col in binary_cols:
        X[col] = X[col].astype(int)
        
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Save model and encoders
    joblib.dump(model, os.path.join(models_dir, 'model.pkl'))
    joblib.dump(label_encoders, os.path.join(models_dir, 'label_encoders.pkl'))
    joblib.dump(feature_columns, os.path.join(models_dir, 'feature_columns.pkl'))
    
    print("Model trained and saved successfully!")

if __name__ == '__main__':
    train_and_save_model()
