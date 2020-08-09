#!/usr/bin/python3
import sys
import warnings
warnings.simplefilter(action='ignore')

import keras
from keras.models import Sequential
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.layers import Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.layers import Dense

import numpy

def read_data(filename, sample_fraction=1.0):

    data = numpy.array(pd.read_csv(filename))
    X = data[:, :-1]
    y = data[:, -1]

    # The entire dataset is 60k images, we can subsample here for quicker testing.
    if sample_fraction < 1.0:
        foo, X, bar, y = train_test_split(X, y, test_size=sample_fraction)


    # The data, split between train and test sets
    X = X.reshape((X.shape[0], X.shape[1], 1))
    X = X.astype('float32')
    X /= 255
    return  (X, y)


def define_model():
    model = Sequential()
    layers = [1, 50, 100, 1]
    model.add(LSTM(
        layers[1],
        input_shape=(None, 1),
        return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(
        layers[2],
        return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(
        layers[3]))
    model.add(Activation("linear"))
    model.compile(loss="mse", optimizer=keras.optimizers.Adam(), metrics=['mae'])
    return model

def centralized_learn(model, trainX, trainY, testX, testY, file):
    bs=32
    model.fit(trainX, trainY, epochs = 1, batch_size=bs, verbose=0)
    _, mae = model.evaluate(testX, testY, verbose=0)
    print(mae)
    file.write("%f\n"%mae)

def run(num):
    trainx, trainy = read_data("/home/sariel/PycharmProjects/fedml/FedML/data/new_train.csv", sample_fraction=1)
    testx, testy = read_data("/home/sariel/PycharmProjects/fedml/FedML/data/data1/test.csv", sample_fraction=1)
    model = define_model()
    output = open("central_result.txt", "a")
    for i in range(num):
        centralized_learn(model, trainx, trainy, testx, testy, output)
    ##fed_learn(trainX, trainY, testX, testY)
    output.close()
run(200)