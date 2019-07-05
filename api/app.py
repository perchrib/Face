from flask import request, render_template 
import flask
from keras.models import load_model as k_load_model
from helpers.constants import Path
from preprocessor.image_processor import get_decoded_base64_img, feature_extract_image, load_model, prepare_image, load_image
import os
import numpy as np

app = flask.Flask(__name__)
app.config["DEBUG"] = True

face_model = None 
image_feature_model = None 

def load_models():
    print("Loading models...")
    global face_model
    global image_feature_model
    face_model = k_load_model(os.path.join(Path.models, "model.h5"))
    face_model._make_predict_function()
    
    image_feature_model = load_model()
    image_feature_model._make_predict_function()

@app.route('/', methods=['GET'])
def home():
    return render_template("camera.html")

@app.route('/predict', methods=['POST'])
def predict():
   
    req_data = request.get_json()
    base64_image = req_data['image'].replace("data:image/png;base64,", "")

    image = get_decoded_base64_img(base64_image)
    image = prepare_image(image, 299)
    feature_image = feature_extract_image(image, image_feature_model)
    print(feature_image)
    print(feature_image.shape)
    label = ["Per", "Liv"]

    prediction = face_model.predict(np.asarray([feature_image]))[0]
    prediction_index = np.argmax(prediction)
    return label[prediction_index]


#if __name__ == "__main__":  
load_models()
app.run()