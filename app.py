from flask import Flask, request, jsonify, render_template
import pickle
import os
import numpy as np

app = Flask(__name__)

# Set the path to the saved pickle file
model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')

# Load the saved model
with open(model_path, 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    #return 'Hello World'
    return render_template('home.html')
    #return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the features from the request
    features = request.get_json(force=True)

    # Convert the features to a list
    features_list = [features['Age'], features['BMI'], features['BloodPressure'],features['Glucose'], features['Diabetics pedigree Function']]

    # Make a prediction using the model
    prediction = model.predict([features_list])[0]

    # Return the prediction as JSON
    return jsonify({'prediction': int(prediction)})

@app.route('/results', methods=['POST'])
def results():
    # get the features from the form submission
    feature1 = request.form['Age']
    feature2 = request.form['BMI']
    feature3 = request.form['BloodPressure']
    feature4 = request.form['Glucose']
    feature5 = request.form['Diabetics pedigree Function']
    
    # load the model and make a prediction based on the user's input features
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    prediction = model.predict([[feature1, feature2, feature3, feature4, feature5]])
    
    # pass the prediction to the results template
    return render_template('result.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)