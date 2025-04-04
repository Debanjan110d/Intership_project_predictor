import joblib
import pandas as pd

def get_input_options():
    """Define valid options and ranges for user input"""
    options = {
        'Gender': ['Female', 'Male', 'Bigender', 'Agender', 'Genderfluid', 'Non-binary', 'Polygender'],
        'Income_Level': ['High', 'Middle', 'Low'],
        'Purchase_Category': [
            'Electronics', 'Sports & Outdoors', 'Jewelry & Accessories', 'Home Appliances', 
            'Toys & Games', 'Mobile Accessories', 'Food & Beverages', 'Beauty & Personal Care', 
            'Books', 'Furniture', 'Health Care', 'Gardening & Outdoors', 'Clothing', 'Luxury Goods',
            'Office Supplies', 'Arts & Crafts', 'Baby Products', 'Animal Feed', 'Travel & Leisure (Flights)',
            'Hotels', 'Health Supplements', 'Software & Apps', 'Groceries', 'Packages'
        ],
        'Device_Used_for_Shopping': ['Desktop', 'Tablet', 'Smartphone']
    }
    
    numeric_ranges = {
        'Age': (18, 50),
        'Purchase_Amount': (50.0, 500.0),
        'Time_Spent_on_Product_Research(hours)': (0, 2.5)
    }
    
    return options, numeric_ranges

def get_user_input():
    """Get input from user with validation"""
    options, numeric_ranges = get_input_options()
    print("\nPlease provide the following information:")
    
    user_input = {}
    
    # Numeric inputs
    for field, (min_val, max_val) in numeric_ranges.items():
        while True:
            try:
                value = float(input(f"\n{field} (range: {min_val} - {max_val}): "))
                if min_val <= value <= max_val:
                    user_input[field] = value
                    break
                print(f"Please enter a value between {min_val} and {max_val}")
            except ValueError:
                print("Please enter a valid number")
    
    # Categorical inputs
    for field, valid_options in options.items():
        print(f"\nAvailable options for {field}:")
        for i, option in enumerate(valid_options, 1):
            print(f"{i}. {option}")
        
        while True:
            try:
                choice = int(input(f"\nSelect {field} (1-{len(valid_options)}): "))
                if 1 <= choice <= len(valid_options):
                    user_input[field] = valid_options[choice-1]
                    break
                print(f"Please enter a number between 1 and {len(valid_options)}")
            except ValueError:
                print("Please enter a valid number")
    
    # Boolean inputs
    for field in ['Discount_Used', 'Customer_Loyalty_Program_Member']:
        while True:
            choice = input(f"\n{field} (yes/no): ").lower()
            if choice in ['yes', 'no']:
                user_input[field] = True if choice == 'yes' else False
                break
            print("Please enter 'yes' or 'no'")
    
    return user_input

def load_model(path='model/'):
    """Load the trained model and transformers"""
    model = joblib.load(f'{path}model.pkl')
    label_encoders = joblib.load(f'{path}label_encoders.pkl')
    feature_columns = joblib.load(f'{path}feature_columns.pkl')
    return model, label_encoders, feature_columns

def prepare_input(data, label_encoders, feature_columns):
    """Prepare input data for prediction"""
    # Create DataFrame with required columns
    input_df = pd.DataFrame([data], columns=feature_columns)
    
    # Encode categorical variables
    for col, le in label_encoders.items():
        if col in input_df.columns:
            input_df[col] = le.transform([str(input_df[col].iloc[0])])
    
    return input_df

def predict_purchase_intent(input_data):
    """Predict purchase intent for given input"""
    # Load model and transformers
    model, label_encoders, feature_columns = load_model()
    
    # Prepare input
    processed_input = prepare_input(input_data, label_encoders, feature_columns)
    
    # Make prediction
    prediction = model.predict(processed_input)
    probability = model.predict_proba(processed_input)
    
    return prediction[0], probability[0]

def display_prediction(prediction, probability):
    """Display prediction results in a user-friendly format"""
    print("\nPrediction Results:")
    print(f"Predicted Purchase Intent: {prediction}")
    print(f"Prediction Probabilities: {probability}")

def main():
    print("Welcome to Purchase Intent Predictor!")
    
    while True:
        user_input = get_user_input()
        
        # Make prediction
        prediction, probability = predict_purchase_intent(user_input)
        
        # Display results
        display_prediction(prediction, probability)
        
        # Ask if user wants to make another prediction
        if input("\nMake another prediction? (yes/no): ").lower() != 'yes':
            break

if __name__ == "__main__":
    main()
