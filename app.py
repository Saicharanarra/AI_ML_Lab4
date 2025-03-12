from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load the model
with open('fish_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the dataset for dummy columns (to match columns during prediction)
df = pd.read_csv("Fish.csv")

# One-hot encoding the 'Species' column to get the dummy columns used during training
df_dummies = pd.get_dummies(df['Species'], drop_first=True)
dummy_columns = df_dummies.columns.tolist()

# List of the columns used for model training, including dummy columns and other numerical features
input_columns = ['Length1', 'Length2', 'Length3', 'Height', 'Width'] + dummy_columns

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect form data
        data = {
            'Length1': [float(request.form['Length1'])],
            'Length2': [float(request.form['Length2'])],
            'Length3': [float(request.form['Length3'])],
            'Height': [float(request.form['Height'])],
            'Width': [float(request.form['Width'])],
        }

        # Create a DataFrame from the input data
        df_input = pd.DataFrame(data)

        # Add missing dummy columns (initialize with 0 for missing species)
        for col in dummy_columns:
            if col not in df_input.columns:
                df_input[col] = 0

        # Ensure that columns match the trained model's input
        df_input = df_input[input_columns]

        # Prediction
        prediction = model.predict(df_input)

        # Return the prediction as JSON
        return jsonify({'Prediction': prediction[0]})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0")
