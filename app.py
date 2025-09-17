from multiprocessing.util import debug
import pickle
from flask import Flask , request,jsonify,render_template
import numpy as np
import pandas as pd
import sklearn


app=Flask(__name__)




model=pickle.load(open('models/model.pkl','rb'))
scaler_md=pickle.load(open('models/scaler.pkl','rb'))

@app.route('/')
def hello():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method=='POST':
            age = int(request.form.get('age'))
            sex = int(request.form.get('sex'))
            cp = int(request.form.get('cp'))
            trestbps = float(request.form.get('trestbps'))
            chol = float(request.form.get('chol'))
            fbs = int(request.form.get('fbs'))
            restecg = int(request.form.get('restecg'))
            thalach = float(request.form.get('thalach'))
            exang = int(request.form.get('exang'))
            oldpeak = float(request.form.get('oldpeak'))
            slope = int(request.form.get('slope'))
            ca = int(request.form.get('ca'))
            thal = int(request.form.get('thal'))
            

            input_data = scaler_md.transform([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
            result=model.predict(input_data)
            
    return render_template('home.html',result=result[0])

        


if __name__=='__main__':
    app.run(debug=True)