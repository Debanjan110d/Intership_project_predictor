# Model Training Documentation
[← Back to README](../README.md) | [Django Deployment Guide →](django_deployment.md)

## Table of Contents
- [Data Overview](#data-overview)
- [Feature Engineering](#feature-engineering)
- [Model Architecture](#model-architecture)
- [Training Process](#training-process)
- [Model Performance](#model-performance)
- [Rule-Based System](#rule-based-system)

## Data Overview
The model uses customer behavior data with the following key features:
- Demographics (Age, Gender)
- Purchase Details (Amount, Category)
- Behavioral Metrics (Research Time, Device Used)
- Customer Status (Loyalty Program, Discount Usage)

### Data Distribution
![Feature Distributions](images/feature_distributions.png)

## Feature Engineering
```python
categorical_cols = ['Gender', 'Income_Level', 'Purchase_Category', 'Device_Used_for_Shopping']
numeric_cols = ['Age', 'Purchase_Amount', 'Time_Spent_on_Product_Research(hours)']
binary_cols = ['Discount_Used', 'Customer_Loyalty_Program_Member']
```

## Model Architecture
- Algorithm: Random Forest Classifier
- Estimators: 100
- Features: 9 input features
- Output: 4 purchase intent categories

### Model Parameters
```python
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
```

## Training Process
1. Data preprocessing
2. Label encoding for categorical variables
3. Model training
4. Rule extraction for interpretability

### Training Code Snippet
```python
# Initialize label encoders
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le
```

## Model Training Process
![Feature Distribution Analysis](images/feature_distributions.png)
*Distribution of key features in training data*

### Data Pipeline
```python
data_preprocessing():
    1. Load data from data.csv
    2. Handle missing values
    3. Encode categorical variables
    4. Scale numerical features
```

### Model Selection
We evaluated multiple algorithms:
- Random Forest (Selected ✓)
- Gradient Boosting
- SVM
- Neural Networks

![Model Performance](images/confusion_matrix.png)
*Confusion Matrix showing model performance*

## Model Performance
- Accuracy: 85%
- F1 Score: 0.83
- Precision: 0.84
- Recall: 0.82

### Confusion Matrix
![Confusion Matrix](images/confusion_matrix.png)

## Rule-Based System
The model is converted to interpretable rules:

```json
{
    "rules": [
        {
            "condition": {
                "field": "Purchase_Amount",
                "operator": "greater_than",
                "value": 400
            },
            "prediction": "Planned",
            "probabilities": {
                "Planned": 70,
                "Impulsive": 10,
                "Wants-based": 15,
                "Need-based": 5
            }
        }
    ]
}
```

For detailed implementation, see [train_model.py](../purchase_predictor/predictor/train_model.py)

## Next Steps
See the [Django Deployment Guide](django_deployment.md) for implementing the web interface.
