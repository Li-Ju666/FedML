#!/usr/bin/python3

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from matplotlib import pyplot
from keras.datasets import fashion_mnist
from keras.utils import to_categorical
from numpy import mean
from numpy import std
from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

def load_dataset(): 
    (trainX, trainY), (testX, testY) = fashion_mnist.load_data()

    trainX = trainX.reshape((trainX.shape[0], 28*28))
    testX = testX.reshape((testX.shape[0], 28*28))

    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    
    print(trainX.shape)
    return trainX, trainY, testX, testY

def prep_pixel(train, test): 
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
    train_norm = train_norm/255.0
    test_norm = test_norm/255.0
    return train_norm, test_norm

def define_model():
    model = Sequential()
    model.add(Dense(100, input_dim = 28*28, activation = 'relu', kernel_initializer='he_uniform'))
    model.add(Dense(100, activation = 'relu'))
    model.add(Dense(10, activation = 'softmax'))

    opt = SGD(lr = 0.01, momentum = 0.9)
    model.compile(optimizer = opt, loss = 'categorical_crossentropy', 
                metrics = ['accuracy'])
    return model

def evaluate_model(dataX, dataY, n_folds = 5): 
    scores, histories = list(), list()
    kfold = KFold(n_folds, shuffle = True, random_state = 1)

    for train_ix, test_ix in kfold.split(dataX): 
        print("start to train")
        model = define_model()
        trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]

        history = model.fit(trainX, trainY, epochs = 10, 
                batch_size = 32, validation_data = (testX, testY), verbose = 0)
        _, acc = model.evaluate(testX, testY, verbose = 0)
        print("> %.3f" % (acc*100.0))
        scores.append(acc)
        histories.append(history)
    return scores, histories

def summarize_diagnostics(histories): 
    for i in range(len(histories)): 
        pyplot.subplot(211)
        pyplot.title('Cross Entropy Loss')
        pyplot.plot(histories[i].history['loss'], color = 'blue', label = 'train')
        pyplot.plot(histories[i].history['val_loss'], color = 'orange', label = 'test')

        pyplot.subplot(212)
        pyplot.title('Classification Accuracy')
        pyplot.plot(histories[i].history['accuracy'], color = 'blue', label = 'train')
        pyplot.plot(histories[i].history['val_accuracy'], color = 'orange', label = 'test')

        pyplot.savefig("diagnosis.png")

def summarize_performance(scores): 
    print('Accuracy: mean = %.3f std = %.3f n = %d' % (mean(scores)*100, std(scores)*100, len(scores)))
    pyplot.boxplot(scores)
    pyplot.savefig('performance.png')

# def run_test_harness(): 
#     trainX, trainY, testX, testY = load_dataset()
#     train_X, test_X = prep_pixel(trainX, testX)
#     scores, histories = evaluate_model(trainX, trainY)

#     summarize_diagnostics(histories)
#     summarize_performance(scores)
def run_test_harness():
    trainX, trainY, testX, testY = load_dataset()
    trainX, testX = prep_pixel(trainX, testX)
    scores, histories = evaluate_model(trainX, trainY)
    summarize_diagnostics(histories)
    summarize_performance(scores)

run_test_harness()
