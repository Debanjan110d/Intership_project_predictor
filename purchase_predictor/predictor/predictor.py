import pandas as pd
from sklearn.preprocessing import LabelEncoder

def prepare_input(form_data):
    # Create a DataFrame with exactly the same column names as training data
    input_df = pd.DataFrame({
        'Age': [form_data['age']],
        'Gender': [form_data['gender']],
        'Device_Used_for_Shopping': [form_data['device_used']],
        'Education_Level': [form_data['education']]
    })
    
    # Encode categorical variables
    le = LabelEncoder()
    categorical_columns = ['Gender', 'Device_Used_for_Shopping', 'Education_Level']
    
    for column in categorical_columns:
        input_df[column] = le.fit_transform(input_df[column])
        
    return input_df

def predict_purchase(model, form_data):
    # Prepare input data
    input_data = prepare_input(form_data)
    
    # Make prediction
    prediction = model.predict(input_data)
    
    return prediction[0]
