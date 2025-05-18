import numpy as np
import pandas as pd
from flask import Flask, request, render_template
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

app = Flask(__name__)

# Load model
lr = joblib.load('diabetes_model_lr.pkl')  

# Load and preprocess dataset
dataset = pd.read_csv('balanced_1500_diabetes_data.csv')
X = dataset[['Glucose', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']].values
y = dataset['Outcome'].values

sc = MinMaxScaler(feature_range=(0, 1))
X_scaled = sc.fit_transform(X)

# Predictions for evaluation
y_pred = lr.predict(X_scaled)

# Calculate metrics
accuracy = f"{accuracy_score(y, y_pred) * 100:.2f}%"
precision = f"{precision_score(y, y_pred) * 100:.2f}%"
recall = f"{recall_score(y, y_pred) * 100:.2f}%"
f1 = f"{f1_score(y, y_pred) * 100:.2f}%"


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    float_features = [float(x) for x in request.form.values()]
    input_array = np.array([float_features])
    input_scaled = sc.transform(input_array)

    probs = lr.predict_proba(input_scaled)[:, 1]
    threshold = 0.5
    prediction = int(probs[0] >= threshold)

    if prediction == 1:
        result_text = f"You have Diabetes"
    else:
        result_text = f"You don't have Diabetes"

    return render_template('index.html', 
                           prediction_text=result_text,
                           accuracy=accuracy,
                           precision=precision,
                           recall=recall,
                           f1=f1)

if __name__ == "__main__":
    app.run(debug=True)
