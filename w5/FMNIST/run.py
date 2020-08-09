#!/usr/bin/python3
import sys
import warnings
warnings.simplefilter(action='ignore')

import keras
from keras.models import Sequential
from keras.optimizers import SGD
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.layers import Conv2D, Dropout
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten

import numpy

def read_data(filename, sample_fraction=1.0):

    data = numpy.array(pd.read_csv(filename))
    X = data[:, 1::]
    y = data[:, 0]

    # The entire dataset is 60k images, we can subsample here for quicker testing.
    if sample_fraction < 1.0:
        foo, X, bar, y = train_test_split(X, y, test_size=sample_fraction)


    # The data, split between train and test sets
    X = X.reshape((X.shape[0], 28, 28, 1))
    X = X.astype('float32')
    X /= 255
    y = to_categorical(y)
    return  (X, y)


def define_model():
    img_rows, img_cols = 28, 28
    input_shape = (img_rows, img_cols, 1)
    num_classes = 10

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])
    return model

def centralized_learn(model, trainX, trainY, testX, testY, file):
    bs=32
    model.fit(trainX, trainY, epochs = 1, batch_size=bs, verbose=0)
    _, accuracy = model.evaluate(testX, testY, verbose=0)
    print(accuracy)
    file.write("%f\n"%accuracy)

def run(num):
    trainx, trainy = read_data("/FedML/data/raw_data.csv", sample_fraction=1)
    testx, testy = testX, testY = read_data("/home/sariel/PycharmProjects/fedml/FedML/data/data1/test.csv", sample_fraction=1)
    model = define_model()
    output = open("central_result.txt", "a")
    for i in range(num):
        _, trainX, _, trainY = train_test_split(trainx, trainy, test_size=0.1)
        _, testX, _, testY = train_test_split(testx, testy, test_size=0.1)
        centralized_learn(model, trainX, trainY, testX, testY, output)
    ##fed_learn(trainX, trainY, testX, testY)
    output.close()
run(50)