import pandas as pd
import numpy as np
import xgboost
import pickle
import os
from flask import Flask,render_template,request
app=Flask(__name__)
model = pickle.load(open('xg_model.pkl', 'rb'))
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict',methods=["POST","GET"])
def predict():
    input_feature=[float(x) for x in request.form.values()]
    features_values=[np.array(input_feature)]
    names= [['playerId','Sex','Equipment','Age','BodyweightKg','BestSquatKg','BestBenchKg']]
    data=pd.DataFrame(features_values,columns=names)
    prediction=model.predict(data)
    print(prediction)
    text="Estimated Deadlift for the builder is:"
    return render_template("index.html",prediction_text=text + str(prediction))
if __name__ == '__main__':
    app.run(debug=False)