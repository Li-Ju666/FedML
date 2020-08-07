import numpy
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

import keras
import numpy

def read_data(filename, sample_fraction=1.0):
    """ Helper function to read and preprocess data for training with Keras. """

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

