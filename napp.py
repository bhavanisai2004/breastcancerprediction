# Importing essential libraries
from flask import Flask, render_template, request
import joblib
import numpy as np


model = joblib.load('brest_cancer.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('newinput.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Extracting features from the form data
        texture_mean = float(request.form['texture_mean'])
        smoothness_mean = float(request.form['smoothness_mean'])
        compactness_mean = float(request.form['compactness_mean'])
        concave_points_mean = float(request.form['concave_points_mean'])
        symmetry_mean = float(request.form['symmetry_mean'])
        fractal_dimension_mean = float(request.form['fractal_dimension_mean'])
        texture_se = float(request.form['texture_se'])
        area_se = float(request.form['area_se'])
        smoothness_se = float(request.form['smoothness_se'])
        compactness_se = float(request.form['compactness_se'])
        concavity_se = float(request.form['concavity_se'])
        concave_points_se = float(request.form['concave_points_se'])
        symmetry_se = float(request.form['symmetry_se'])
        fractal_dimension_se = float(request.form['fractal_dimension_se'])
        texture_worst = float(request.form['texture_worst'])
        area_worst = float(request.form['area_worst'])
        smoothness_worst = float(request.form['smoothness_worst'])
        compactness_worst = float(request.form['compactness_worst'])
        concavity_worst = float(request.form['concavity_worst'])
        concave_points_worst = float(request.form['concave_points_worst'])
        symmetry_worst = float(request.form['symmetry_worst'])
        fractal_dimension_worst = float(request.form['fractal_dimension_worst'])

        # Create an array of features
        data = np.array([[texture_mean, smoothness_mean, compactness_mean, concave_points_mean,
                          symmetry_mean, fractal_dimension_mean, texture_se, area_se, smoothness_se,
                          compactness_se, concavity_se, concave_points_se, symmetry_se,
                          fractal_dimension_se, texture_worst, area_worst, smoothness_worst,
                          compactness_worst, concavity_worst, concave_points_worst, symmetry_worst,
                          fractal_dimension_worst]])

        # Making prediction
        my_prediction = model.predict(data)

        # Determine suggestion based on prediction
        if my_prediction == 1:
            suggestion = "It is recommended to consult with a healthcare professional for further evaluation and possible treatment options."
        else:
            suggestion = "No further action is required at this time, but regular screenings are recommended."

        # Returning the prediction and suggestion as response
        return render_template('newresult.html', prediction=my_prediction, suggestion=suggestion)

if __name__ == '__main__':
    app.run(debug=True)
