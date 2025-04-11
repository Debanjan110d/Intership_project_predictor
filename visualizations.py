import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def create_visualizations(data_path='data.csv'):
    # Load data
    df = pd.read_csv(data_path)
    
    # 1. Correlation Heatmap
    plt.figure(figsize=(12, 8))
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    correlation_matrix = df[numerical_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap of Numerical Features')
    plt.tight_layout()
    plt.savefig('plots/correlation_heatmap.png')
    plt.close()

    # 2. Distribution Plot for Age
    plt.figure(figsize=(10, 6))
    sns.displot(data=df, x='Age', kind='kde', fill=True)
    plt.title('Distribution of Customer Ages')
    plt.savefig('plots/age_distribution.png')
    plt.close()

    # 3. Purchase Amount Histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='Purchase_Amount', bins=30)
    plt.title('Distribution of Purchase Amounts')
    plt.xlabel('Purchase Amount')
    plt.ylabel('Frequency')
    plt.savefig('plots/purchase_amount_histogram.png')
    plt.close()

    # 4. Feature Importance Barplot
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder

    # Prepare data for feature importance
    X = df.drop(['Purchase_Intent', 'Customer_ID'], axis=1)
    y = df['Purchase_Intent']

    # Encode categorical variables
    for column in X.select_dtypes(include=['object']):
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column])

    # Train a simple model to get feature importance
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Create feature importance plot
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=True)

    plt.figure(figsize=(10, 8))
    sns.barplot(data=feature_importance, x='Importance', y='Feature')
    plt.title('Feature Importance for Purchase Intent Prediction')
    plt.tight_layout()
    plt.savefig('plots/feature_importance.png')
    plt.close()

if __name__ == "__main__":
    # Create plots directory if it doesn't exist
    import os
    if not os.path.exists('plots'):
        os.makedirs('plots')
    
    create_visualizations()
    print("Visualizations have been created in the 'plots' directory!")
