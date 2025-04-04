import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def analyze_dataset(data_path):
    """Analyze the dataset and print important statistics"""
    df = pd.read_csv(data_path)
    
    print("Dataset Shape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    
    # Identify numeric and categorical columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    print("\nNumeric columns:", len(numeric_cols))
    print("Categorical columns:", len(categorical_cols))
    
    # Basic statistics for numeric columns
    print("\nNumeric Statistics:")
    print(df[numeric_cols].describe())
    
    # Value counts for categorical columns
    print("\nCategory Distributions:")
    for col in categorical_cols:
        print(f"\n{col}:")
        print(df[col].value_counts().head())
    
    return numeric_cols, categorical_cols

if __name__ == "__main__":
    analyze_dataset('data.csv')
