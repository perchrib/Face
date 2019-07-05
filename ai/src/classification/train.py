from helpers.io import load_pickle
from helpers.constants import Path
from preprocessor.dataset_processor import split_data
#
import numpy as np
import matplotlib.pyplot as plt
import os
#keras
from keras.layers import Dense, Dropout, Activation
from keras.models import Model, Sequential
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping

def init_model():
    print('Init Model ... ')
    model = Sequential()
    model.add(Dense(1024, input_dim=1536, init='uniform'))
    model.add(Activation('tanh'))
    model.add(Dropout(0.4))
    model.add(Dense(750, init='uniform'))
    model.add(Activation('tanh'))
    model.add(Dropout(0.4))
    model.add(Dense(512, init='uniform'))
    model.add(Activation('tanh'))
    model.add(Dropout(0.4))
    model.add(Dense(300, init='uniform'))
    model.add(Activation('tanh'))
    model.add(Dense(2, activation='softmax'))
    rms = RMSprop(lr=0.0001)
    model.compile(loss='categorical_crossentropy', optimizer=rms, metrics=['accuracy'])
    print('Model compiled')
    return model

def plot_result(history):
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

if __name__ == "__main__":
    train_data = load_pickle(Path.train, "train.pickle")
    validate_data = load_pickle(Path.validate, "validate.pickle")
    
    x_train, y_train = split_data(train_data)
    x_val, y_val = split_data(validate_data)
    print("X-shape: ", x_train.shape, "Y-shape: ", y_train.shape)
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    model = init_model()
    history = model.fit(x_train, y_train, nb_epoch=100, batch_size=128,
              callbacks=[early_stopping],
              validation_data=(x_val, y_val), verbose=1)

    model.save(os.path.join(Path.models, "model.h5"))
    plot_result(history)


