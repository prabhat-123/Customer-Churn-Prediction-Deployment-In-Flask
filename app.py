import tensorflow as tf
from keras.models import Sequential,load_model,model_from_json
from keras.layers import Dense, Dropout,Activation,MaxPooling2D,Conv2D,Flatten
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.preprocessing.image import load_img
from keras.preprocessing import image
import numpy as np
import h5py
import os
import sys
import json
from sklearn.preprocessing import StandardScaler
from predictor import sc

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename


# Define a flask app
app = Flask(__name__)

with open('customer_churn_prediction_model.json','r') as f:
    model = model_from_json(f.read())


# Load your trained model
model.load_weights('customer_churn_prediction_model.h5')   

print('Model loaded. Check http://127.0.0.1:5000/')

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('prediction.html')


@app.route('/', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the values from the form 
        credit_score = request.form['cr_score']
        age = request.form['age']
        tenure = request.form['tenure']
        balance = request.form.get('balance')
        number_of_products = request.form.get('no_of_products')
        estimated_salary = request.form['salary']
        country = request.form['country']
        gender = request.form['gender']
        has_credit_card = request.form['cr_card']
        is_active_member = request.form['active_member']
        print([credit_score,age,tenure,balance,number_of_products,estimated_salary,country,gender,has_credit_card,is_active_member])
        # Process input 
        if country=="France":
            countries= [0,0]
        elif country=="Germany":
            countries = [1,0]
        else:
            countries = [0,1]
        # Make Prediction
        prediction = model.predict(sc.transform(np.array([[countries[0],countries[1],credit_score,gender,age,tenure,balance,number_of_products,has_credit_card,is_active_member,estimated_salary]])))
        # Process your result for human
        if prediction > 0.5:
            result = "The customer will leave the bank"
        else:
            result = "The customer won't leave the bank"
        return result
    return None

        


if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=5000)
