from flask import Flask, request, render_template
import pandas as pd
import pickle

app = Flask(__name__)

# Load model, imputer, and feature columns once when the app starts
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('imputer.pkl', 'rb') as f:
    imputer = pickle.load(f)

with open('feature_columns.pkl', 'rb') as f:
    feature_cols = pickle.load(f)


@app.route('/')
def home():
    return render_template('index.html')  # Your input form HTML page


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract inputs from form
        input_dict = {
            'Pclass': int(request.form['Pclass']),
            'Sex': request.form['Sex'],
            'Age': float(request.form['Age']),
            'SibSp': int(request.form['SibSp']),
            'Parch': int(request.form['Parch']),
            'Fare': float(request.form['Fare']),
            'Embarked': request.form['Embarked']
        }

        # Convert to DataFrame
        input_df = pd.DataFrame([input_dict])

        # One-hot encode categorical columns (Sex, Embarked)
        input_encoded = pd.get_dummies(input_df)

        # Add missing columns (from training) with zeros
        for col in feature_cols:
            if col not in input_encoded.columns:
                input_encoded[col] = 0

        # Ensure the order of columns is same as training
        input_encoded = input_encoded[feature_cols]

        # Impute missing numeric values (if any)
        input_imputed = imputer.transform(input_encoded)

        # Predict survival
        prediction = model.predict(input_imputed)[0]

        result = 'Survived' if prediction == 1 else 'Did not survive'

        return render_template('result.html', prediction=result)

    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == '__main__':
    app.run(debug=True)
