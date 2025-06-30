from flask import Flask, request, jsonify
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

app = Flask(__name__)

# Load and train model
X, y = load_iris(return_X_y=True)
model = LogisticRegression(max_iter=200).fit(X, y)

# Define prediction route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['features']
    prediction = model.predict([data])
    return jsonify({'prediction': int(prediction[0])})

# Define home route
@app.route('/')
def home():
    return "Flask server running"

if __name__ == '__main__':
    app.run(debug=True)
