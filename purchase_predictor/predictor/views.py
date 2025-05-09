from django.shortcuts import render, redirect
from django.contrib import messages
from .forms import PredictionForm
import json
import os

def predict(request):
    if request.method == 'POST':
        try:
            form = PredictionForm(request.POST)
            if form.is_valid():
                # Get input data from form
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

                # Ensure models directory exists
                model_path = os.path.join(os.path.dirname(__file__), 'models')
                os.makedirs(model_path, exist_ok=True)
                
                model_file = os.path.join(model_path, 'model.json')
                
                # Create default model file if it doesn't exist
                if not os.path.exists(model_file):
                    default_model = {
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
                            },
                            {
                                "condition": {
                                    "field": "Time_Spent_on_Product_Research(hours)",
                                    "operator": "less_than",
                                    "value": 1
                                },
                                "prediction": "Impulsive",
                                "probabilities": {
                                    "Planned": 10,
                                    "Impulsive": 65,
                                    "Wants-based": 15,
                                    "Need-based": 10
                                }
                            },
                            {
                                "condition": {
                                    "field": "Discount_Used",
                                    "operator": "equals",
                                    "value": True
                                },
                                "prediction": "Wants-based",
                                "probabilities": {
                                    "Planned": 15,
                                    "Impulsive": 20,
                                    "Wants-based": 55,
                                    "Need-based": 10
                                }
                            }
                        ]
                    }
                    with open(model_file, 'w') as f:
                        json.dump(default_model, f)

                # Load model and make prediction
                with open(model_file, 'r') as f:
                    model_data = json.load(f)
                    
                # Make prediction using the model data
                prediction = predict_purchase_intent(input_data, model_data)
                
                # Store prediction data in session
                request.session['prediction_data'] = {
                    'prediction': prediction[0],
                    'probabilities': prediction[1],
                    'input_data': input_data
                }
                
                return redirect('result')
            else:
                messages.error(request, "Please correct the form errors.")
                return render(request, 'predictor/predict.html', {'form': form})
                
        except Exception as e:
            messages.error(request, f'Error: {str(e)}')
            return render(request, 'predictor/predict.html', {'form': PredictionForm()})
    else:
        form = PredictionForm()

    return render(request, 'predictor/predict.html', {'form': form})

def result(request):
    prediction_data = request.session.get('prediction_data')
    if not prediction_data:
        return redirect('predict')
    return render(request, 'predictor/result.html', prediction_data)

def predict_purchase_intent(input_data, model_data):
    # Initialize scores for each category
    scores = {
        'Planned': 0,
        'Impulsive': 0,
        'Wants-based': 0,
        'Need-based': 0
    }
    
    # Check each rule and accumulate scores
    matches = 0
    for rule in model_data['rules']:
        if evaluate_rule(input_data, rule['condition']):
            matches += 1
            if 'probabilities' in rule:
                for intent, prob in rule['probabilities'].items():
                    scores[intent] += prob
            else:
                scores[rule['prediction']] += 100

    # If no rules matched, use default logic
    if matches == 0:
        default_pred = determine_default_prediction(input_data)
        scores[default_pred] = 100
    else:
        # Normalize scores
        total = sum(scores.values())
        if total > 0:
            for intent in scores:
                scores[intent] = round((scores[intent] / total) * 100)

    # Get prediction (highest scoring intent)
    prediction = max(scores.items(), key=lambda x: x[1])[0]
    
    # Return both prediction and probabilities
    return prediction, scores

def determine_default_prediction(input_data):
    amount = float(input_data['Purchase_Amount'])
    research_time = float(input_data['Time_Spent_on_Product_Research(hours)'])
    loyalty = input_data['Customer_Loyalty_Program_Member']
    
    # More sophisticated default logic
    if amount > 400 and research_time > 1.5:
        return 'Planned'
    elif amount < 100 and research_time < 0.5:
        return 'Impulsive'
    elif loyalty and input_data['Discount_Used']:
        return 'Wants-based'
    else:
        return 'Need-based'

def evaluate_rule(input_data, condition):
    # Evaluate a single rule condition
    field = condition['field']
    op = condition['operator']
    value = condition['value']
    
    if field not in input_data:
        return False
        
    if op == 'equals':
        return input_data[field] == value
    elif op == 'greater_than':
        return float(input_data[field]) > float(value)
    elif op == 'less_than':
        return float(input_data[field]) < float(value)
    return False
