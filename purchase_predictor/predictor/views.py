from django.shortcuts import render, redirect
from django.contrib import messages
from .forms import PredictionForm
import joblib
import pandas as pd
import os
import numpy as np

def predict(request):
    prediction_result = None
    probabilities = None
    input_data = None

    if request.method == 'POST':
        form = PredictionForm(request.POST)
        if form.is_valid():
            try:
                # Prepare input data
                input_data = {
                    'Age': form.cleaned_data['age'],
                    'Gender': form.cleaned_data['gender'],
                    'Income_Level': form.cleaned_data['income_level'],
                    'Purchase_Amount': form.cleaned_data['purchase_amount'],
                    'Purchase_Category': form.cleaned_data['purchase_category'],
                    'Device_Used_for_Shopping': form.cleaned_data['device'],
                    'Time_Spent_on_Product_Research(hours)': form.cleaned_data['time_spent'],
                    'Discount_Used': form.cleaned_data['discount_used'],
                    'Customer_Loyalty_Program_Member': form.cleaned_data['loyalty_member']
                }
                
                # Load model and make prediction
                model_path = os.path.join(os.path.dirname(__file__), 'models')
                model = joblib.load(os.path.join(model_path, 'model.pkl'))
                label_encoders = joblib.load(os.path.join(model_path, 'label_encoders.pkl'))
                feature_columns = joblib.load(os.path.join(model_path, 'feature_columns.pkl'))
                
                # Prepare input data
                input_df = pd.DataFrame([input_data], columns=feature_columns)
                for col, le in label_encoders.items():
                    if col in input_df.columns:
                        input_df[col] = le.transform([str(input_df[col].iloc[0])])
                
                # Make prediction
                prediction_result = model.predict(input_df)[0]
                probabilities = model.predict_proba(input_df)[0]
                
                # Format probabilities for display
                intent_types = ['Impulsive', 'Need-based', 'Planned', 'Wants-based']
                prob_dict = {intent: round(prob * 100, 2) for intent, prob in zip(intent_types, probabilities)}
                
                # Redirect to result page with prediction data
                return render(request, 'predictor/result.html', {
                    'prediction': prediction_result,
                    'probabilities': prob_dict,
                    'input_data': input_data
                })
                
            except Exception as e:
                messages.error(request, f'Prediction error: {str(e)}')
                print(f"Error during prediction: {str(e)}")
    else:
        form = PredictionForm()

    return render(request, 'predictor/predict.html', {
        'form': form
    })
