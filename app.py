# from flask import Flask, request, jsonify, render_template
# import pickle
# import os
# import numpy as np

# app = Flask(__name__)

# # Set the path to the saved pickle file
# model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')

# # Load the saved model
# with open(model_path, 'rb') as f:
#     model = pickle.load(f)

# @app.route('/')
# def home():
#     #return 'Hello World'
#     return render_template('home.html')
#     #return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     # Get the features from the request
#     features = request.get_json(force=True)
#     print(features)

#     # Convert the features to a list
#     features_list = [features['Age'], features['BMI'], features['BloodPressure'],features['Glucose'], features['Diabetics pedigree Function']]

#     # Make a prediction using the model
#     prediction = model.predict([features_list])[0]

#     # # Return the prediction as JSON
#     # return jsonify({'prediction': int(prediction)})
#       #output = round(prediction[0], 2)
#     return render_template('home.html', prediction_text="prediction test {}".format(prediction[0]))


# @app.route('/results', methods=['POST'])
# def results():
#     # get the features from the form submission
#     feature1 = request.form['Age']
#     feature2 = request.form['BMI']
#     feature3 = request.form['BloodPressure']
#     feature4 = request.form['Glucose']
#     feature5 = request.form['Diabetics pedigree Function']
    
#     # load the model and make a prediction based on the user's input features
#     with open('model.pkl', 'rb') as f:
#         model = pickle.load(f)
#     prediction = model.predict([[feature1, feature2, feature3, feature4, feature5]])
    
#     # # pass the prediction to the results template
#     # return render_template('result.html', prediction=prediction)
#     output = prediction[0]
#     return jsonify(output)

# if __name__ == '__main__':
#     app.run(debug=True)

import numpy as np
from flask import Flask, request, jsonify, render_template, url_for
import pickle


app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    #return 'Hello World'
    return render_template('home.html')
    #return render_template('index.html')

@app.route('/predict',methods = ['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    print(prediction[0])

    #output = round(prediction[0], 2)
    return render_template('home.html', prediction_text="Results {}".format(prediction[0]))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)



if __name__ == '__main__':
    app.run(debug=True)