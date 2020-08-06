import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
import os

from scaleout.project import Project

import tempfile

# Create an initial CNN Model
def create_seed_model():
	# input image dimensions
	model = Sequential()
	model.add(Dense(100, input_dim=28 * 28, activation='relu',
					kernel_initializer='he_uniform'))
	model.add(Dense(100, activation='relu'))
	model.add(Dense(10, activation='softmax'))

	opt = SGD(lr=0.01, momentum=0.9)
	model.compile(optimizer=opt, loss='categorical_crossentropy',
				  metrics=['accuracy'])
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

	

