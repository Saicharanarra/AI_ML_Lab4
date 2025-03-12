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
    data = {key: [float(value)] for key, value in request.form.items()}
    df = pd.DataFrame(data)
    prediction = model.predict(df)
    return jsonify({'Prediction': prediction[0]})

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0")
