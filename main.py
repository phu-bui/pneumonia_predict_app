from __future__ import division, print_function
import os
from libs.api_response import ApiResponse
import numpy as np
from flask_jwt_extended import jwt_required
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array

from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = './model/model.h5'

# Load your trained model
model = load_model(MODEL_PATH)
# model._make_predict_function()          # Necessary to make everything ready to run on the GPU ahead of time
print('Model loaded. Start serving...')


# You can also use pretrained model from Keras
# Check https://keras.io/applications/
# from keras.applications.resnet50 import ResNet50
# model = ResNet50(weights='imagenet')
# print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))  # target_size must agree with what the trained model expects!!

    # Preprocessing the image
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    predict = model.predict(images, batch_size=1)
    return predict


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploaded', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        predict = model_predict(file_path, model)
        os.remove(file_path)  # removes file from the server after prediction has been returned

        # Arrange the correct return according to the model.
        # In this model 1 is Pneumonia and 0 is Normal.
        str1 = 'Pneumonia'
        str2 = 'Normal'

        data = {}
        data.update(f)
        if predict[0][0] == 1:
            return str2
        else:
            return str1
    return None


if __name__ == '__main__':
    app.run(debug=True)


#https://github.com/tberris/deep-learning-pneumonia-detection-web-app/blob/master/app.py