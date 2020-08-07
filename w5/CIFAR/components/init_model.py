import keras
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential
import os

from scaleout.project import Project

import tempfile

# Create an initial CNN Model
def create_seed_model():
	model = Sequential()
	model.add(
		Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Flatten())
	model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(10, activation='softmax'))
	# compile model
	model.compile(optimizer=keras.optimizers.Adam(),
				  loss='categorical_crossentropy', metrics=['accuracy'])
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

	

