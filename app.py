from flask import Flask, render_template, request, url_for, redirect
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline


application = Flask(__name__)
app = application

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethinicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=int(request.form.get('reading_score')),
            writing_score=int(request.form.get('writing_score'))
        )
        final_data = data.get_data_as_dataframe()
        print(final_data)
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(final_data)

        return render_template('home.html', results=results)
        
    return render_template('home.html')


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)