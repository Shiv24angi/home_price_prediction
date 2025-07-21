from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle
import numpy as np
# Import all necessary classes that were part of your original pipeline definition
# This is vital for pickle to load the model correctly.
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# --- Load the Trained Model ---
try:
    with open('house_price_model.pkl', 'rb') as file:
        regression_model = pickle.load(file)
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Error: Model file 'house_price_model.pkl' not found. Ensure it's in the same directory as app.py.")
    regression_model = None # Set to None to handle loading errors gracefully
except Exception as e:
    print(f"An unexpected error occurred while loading the model: {e}")
    regression_model = None

# --- Define Expected Columns for Model Input ---
# This is CRUCIAL for consistent predictions.
# You need the exact list of columns your model was trained with,
# especially after one-hot encoding 'furnishingstatus'.
# Based on your notebook's X.head() output after preprocessing (after 'price' column was dropped), the columns are:
EXPECTED_COLUMNS = ['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom',
                    'basement', 'hotwaterheating', 'airconditioning', 'parking', 'prefarea',
                    'furnishingstatus_semi-furnished', 'furnishingstatus_unfurnished']


# --- Flask Routes ---

@app.route('/')
def home():
    """Renders the main HTML page for user interaction."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles POST requests for house price predictions.
    It receives JSON data, preprocesses it, makes a prediction, and returns the result.
    """
    if regression_model is None:
        return jsonify({'error': 'Model not loaded. Please check server logs.'}), 500

    try:
        data = request.json # Get JSON data from the request body
        if not data:
            return jsonify({'error': 'No input data provided'}), 400

        # Create a DataFrame from the incoming JSON data
        # Ensure that numerical values are parsed correctly (Flask's request.json might keep them as strings)
        input_data = {
            'area': float(data.get('area')),
            'bedrooms': int(data.get('bedrooms')),
            'bathrooms': int(data.get('bathrooms')),
            'stories': int(data.get('stories')),
            'parking': int(data.get('parking')),
            'mainroad': data.get('mainroad').lower(), # Convert to lowercase for consistent 'yes'/'no' check
            'guestroom': data.get('guestroom').lower(),
            'basement': data.get('basement').lower(),
            'hotwaterheating': data.get('hotwaterheating').lower(),
            'airconditioning': data.get('airconditioning').lower(),
            'prefarea': data.get('prefarea').lower(),
            'furnishingstatus': data.get('furnishingstatus')
        }
        new_house_df = pd.DataFrame([input_data])

        # --- Preprocessing Steps (MUST match your notebook's preprocessing for X) ---

        # 1. Convert 'yes'/'no' columns to 1/0
        for col in ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']:
            new_house_df[col] = new_house_df[col].apply(lambda x: 1 if x == 'yes' else 0)

        # 2. Apply One-Hot Encoding to 'furnishingstatus'
        # Crucially, `drop_first=True` should match your training.
        # This will create new columns like 'furnishingstatus_semi-furnished', 'furnishingstatus_unfurnished'
        new_house_df_encoded = pd.get_dummies(new_house_df, columns=['furnishingstatus'], drop_first=True)

        # 3. Align columns with the training data (add missing, ensure order)
        # This is very important if the new data doesn't have all possible one-hot encoded columns (e.g., if furnishingstatus was 'furnished' but not 'semi-furnished')
        processed_input_for_model = pd.DataFrame(columns=EXPECTED_COLUMNS) # Create a template DataFrame
        processed_input_for_model = pd.concat([processed_input_for_model, new_house_df_encoded], ignore_index=True)
        processed_input_for_model = processed_input_for_model.fillna(0) # Fill any newly introduced NaN (from missing OHE cols) with 0

        # Ensure column order
        processed_input_for_model = processed_input_for_model[EXPECTED_COLUMNS]

        # Convert boolean columns to integer type if pd.get_dummies created bools
        for col in processed_input_for_model.columns:
            if processed_input_for_model[col].dtype == bool:
                processed_input_for_model[col] = processed_input_for_model[col].astype(int)
        # --- End of Preprocessing ---

        # Make prediction using the loaded pipeline
        prediction = regression_model.predict(processed_input_for_model)[0]

        return jsonify({'predicted_price': round(prediction, 2)}) # Return prediction as JSON

    except Exception as e:
        # Log the error for debugging purposes
        print(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500 # Return an error message to the frontend

if __name__ == '__main__':
    # Run the Flask development server
    # debug=True allows for automatic reloading on code changes and provides a debugger
    # Set to False for production deployments.
    app.run(debug=True)