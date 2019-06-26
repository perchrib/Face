from keras.models import load_model
from helpers.constants import Path
from preprocessor.dataset_processor import split_data
from helpers.io import load_pickle
import os
import numpy as np


def cosine_similarity(vA, vB):
    return np.dot(vA, vB) / (np.linalg.norm(vA) * np.linalg.norm(vB))


if __name__ == "__main__":
    model = load_model(os.path.join(Path.models, "model.h5"))
    ## remove later
    test_data = load_pickle(Path.test, "test.pickle")
    x_test, y_test = split_data(test_data)
    
    if len(x_test) != len(y_test):
        raise Error("List not equal")
    
    for i in range(len(x_test)):
        label = ["per", "liv"]
        x_i = x_test[i]
        y_i = y_test[i]
        prediction = model.predict(np.asarray([x_i]))[0]
        pre_index = np.argmax(prediction)
        pre_other = np.argmin(prediction)
        truth_index = np.argmax(y_i)
        print(y_i, prediction)
        print("Prediction: ", label[pre_index], " Truth: ", label[truth_index], "Accuracy: ", prediction[pre_index], " vs: ", prediction[pre_other])
        # print(i, _i)
        # print(prediction)
    ##

