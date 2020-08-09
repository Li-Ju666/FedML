import keras

from keras.models import Sequential
from keras.layers import Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.layers import Dense
import os

from scaleout.project import Project

import tempfile

# Create an initial CNN Model
def create_seed_model():
	# input image dimensions
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

if __name__ == '__main__':

	# Create a seed model and push to Minio
	model = create_seed_model()
	fod, outfile_name = tempfile.mkstemp(suffix='.h5') 
	model.save(outfile_name)

	project = Project()
	from scaleout.repository.helpers import get_repository
	storage = get_repository(project.config['Alliance']['Repository'])

	model_id = storage.set_model(outfile_name,is_file=True)
	os.unlink(outfile_name)
	print("Created seed model with id: {}".format(model_id))

	

