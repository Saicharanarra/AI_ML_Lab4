from flask import Flask, request, jsonify, render_template
import pickle  
import pandas as pd

app = Flask(__name__)

# Load model
with open('fish_model.pkl', 'rb') as file:
    model = pickle.load(file)

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
            'Width': [float(request.form['Width'])]
        }

        # Create a DataFrame to pass to the model
        df = pd.DataFrame(data)

        # Prediction
        prediction = model.predict(df)

        # Return the prediction as JSON
        return jsonify({'Prediction': prediction[0]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0")
