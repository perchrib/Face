from PIL import Image
import numpy as np
from io import BytesIO
import base64

from keras.preprocessing import image
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_resnet_v2 import preprocess_input
from keras.models import Model

def prepare_image(image, size):
    old_size = image.size
    ratio = float(size)/max(old_size)

    new_size = tuple([int(x*ratio) for x in old_size])
    old_im = image.resize(new_size, Image.ANTIALIAS)
    new_im = Image.new("RGB", (size, size))
    new_im.paste(old_im, ((size - new_size[0]) // 2,
                (size - new_size[1]) // 2))
    
    return new_im

def load_image(path):
    image = Image.open(path)
    return image

def feature_extract_image(img, model):
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    print("Extract image")
    feature = model.predict(x)
    #feature = model._make_predict_function(x)
    return feature[0]

def load_model():
    base_model = InceptionResNetV2(weights='imagenet')
    model = Model(input=base_model.input, output=base_model.get_layer('avg_pool').output)
    return model

def get_decoded_base64_img(base64_string):
    #data['img'] = base64_string
    return Image.open(BytesIO(base64.b64decode(base64_string)))



