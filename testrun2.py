import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv('data.csv')

print(data.head())

print("Missing values per column:\n", data.isnull().sum())

data['Social_Media_Influence'].fillna(data['Social_Media_Influence'].mean(), inplace=True)
data['Engagement_with_Ads'].fillna(data['Engagement_with_Ads'].mean(), inplace=True)

categorical_cols = data.select_dtypes(include=['object']).columns
data[categorical_cols] = data[categorical_cols].apply(lambda x: x.str.lower())

duplicates = data.duplicated().sum()
print(f"Number of duplicate rows: {duplicates}")
data.drop_duplicates(inplace=True)

X = data[['Age', 'Income_Level', 'Purchase_Amount', 'Frequency_of_Purchase',
          'Brand_Loyalty', 'Product_Rating', 'Time_Spent_on_Product_Research(hours)',
          'Social_Media_Influence', 'Discount_Sensitivity', 'Return_Rate', 'Customer_Satisfaction']]

y = data['Purchase_Intent']

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print("Classification Report:\n", classification_report(y_test, y_pred))

test_data = [
    [25, 45000, 200, 5, 8, 4.5, 2.5, 30, 7, 0.2, 9],
    [35, 60000, 350, 7, 9, 4.8, 3.2, 50, 5, 0.1, 8],
    [22, 32000, 150, 3, 6, 4.2, 1.8, 20, 6, 0.3, 7]
]

test_df = pd.DataFrame(test_data, columns=X.columns)
predictions = model.predict(test_df)
predicted_labels = label_encoder.inverse_transform(predictions)

for i, pred in enumerate(predicted_labels):
    print(f"Test Case {i+1}: Predicted Purchase Intent -> {pred.upper()}")

