from flask import Flask, render_template, request
import pandas as pd
import pickle
from sklearn.base import BaseEstimator

app = Flask(__name__)
data = pd.read_csv('final_dataset.csv')
pipe = pickle.load(open("RidgeModel.pkl", 'rb'))

# Function to preprocess input data
def preprocess_input(input_data):
    # Convert 'baths' column to numeric with errors='coerce'
    input_data['baths'] = pd.to_numeric(input_data['baths'], errors='coerce')

    # Convert input data to numeric types
    input_data = input_data.astype({'beds': int, 'baths': float, 'size': float, 'zip_code': int})

    # Handle unknown categories in the input data
    for column in input_data.columns:
        unknown_categories = set(input_data[column]) - set(data[column].unique())
        if unknown_categories:
            print(f"Unknown categories in {column}: {unknown_categories}")
            # Handle unknown categories (e.g., replace with a default value)
            input_data[column] = input_data[column].replace(unknown_categories, data[column].mode()[0])

    return input_data

@app.route('/')
def index():
    bedrooms = sorted(data['beds'].unique())
    bathrooms = sorted(data['baths'].unique())
    sizes = sorted(data['size'].unique())
    zip_codes = sorted(data['zip_code'].unique())

    return render_template('index.html', bedrooms=bedrooms, bathrooms=bathrooms, sizes=sizes, zip_codes=zip_codes)

@app.route('/predict', methods=['POST'])
def predict():
    bedrooms = request.form.get('beds')
    bathrooms = request.form.get('baths')
    size = request.form.get('size')
    zipcode = request.form.get('zip_code')

    # Create a DataFrame with the input data
    input_data = pd.DataFrame([[bedrooms, bathrooms, size, zipcode]],
                               columns=['beds', 'baths', 'size', 'zip_code'])

    print("Input Data:")
    print(input_data)

    # Preprocess the input data
    input_data = preprocess_input(input_data)

    print("Processed Input Data:")
    print(input_data)

    # Predict the price after applying the same transformation as during training
    prediction = pipe.predict(input_data)  # Assuming your model doesn't need further transformation
    prediction = prediction[0]

    return str(prediction)

if __name__ == "__main__":
    # Check the type of the pipe object and whether it has a predict method
    if isinstance(pipe, BaseEstimator) and hasattr(pipe, 'predict'):
        print("Model object type: Scikit-learn model")
        print("The model object has a predict method.")
        app.run(debug=True, port=5001)
    else:
        print("Error: The type of the pipe object is not a scikit-learn model or it doesn't have a predict method.")

# Generate the URL for the Flask application
url = "http://127.0.0.1:5001/"
print("Your Flask application is running at:", url)
