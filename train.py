"""train.py

  This module contains the necessary functionality for training a neural network
    using the audio dataset location.

"""

LearningDatasetLocation = 'E:/FACULTATE/MusicClassification/models/'
SoundwavesDataFileLocation = LearningDatasetLocation + 'songs_data.npy'
GenresDataFileLocation = LearningDatasetLocation + 'genres_data.npy'

SavedModelFileLocation = LearningDatasetLocation + 'model.json'
SavedWeightsFileLocation = LearningDatasetLocation + 'weights.h5'

input_shape = (128, 128)
batch_size = 256
epochs = 100

import os
import gc
import numpy
import keras

import matplotlib.pyplot as pyplot
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from src.Model import AudioModel
from src.Utils import split_songs

if os.path.isfile(SoundwavesDataFileLocation) and os.path.isfile(GenresDataFileLocation):

  soundwaves = numpy.load(SoundwavesDataFileLocation)
  genres = numpy.load(GenresDataFileLocation)

else:
  print('The files containing the soundwaves were not found at path\n' +
        SoundwavesDataFileLocation + '\n' +
        GenresDataFileLocation + '\n' +
        'Please ensure you run createAudioModels.py before training')
  exit(1)

from time import time

inputs_train, inputs_test, outputs_train, outputs_test = train_test_split(
  soundwaves, genres, test_size = 0.1, stratify = genres)

inputs_train, inputs_validation, outputs_train, outputs_validation = train_test_split(
  inputs_train, outputs_train, test_size = 1 / 6, stratify = outputs_train)

inputs_validation, outputs_validation = split_songs(
  inputs_validation, outputs_validation)
inputs_test, outputs_test = split_songs(
  inputs_test, outputs_test)
inputs_train, outputs_train = split_songs(
  inputs_train, outputs_train)

model = AudioModel(input_shape)

adam = keras.optimizers.Adam(lr = 0.001)

model.compile(
  loss = keras.losses.mean_squared_logarithmic_error,
  optimizer = adam,
  metrics = [ 'accuracy' ]
)

reduce_lr = keras.callbacks.ReduceLROnPlateau(
  monitor='val_acc',
  patience=5,
  factor=0.6
)
tensorboard = keras.callbacks.TensorBoard(log_dir="./logs/{}".format(time()))

history = model.fit(
  inputs_train, outputs_train,
  batch_size = batch_size,
  epochs = epochs,
  verbose = 1,
  validation_data = ( inputs_validation, outputs_validation ),
  callbacks = [ reduce_lr, tensorboard ])

score = model.evaluate(
  inputs_test, outputs_test
)
score_validation = model.evaluate(
  inputs_validation, outputs_validation
)


with open(SavedModelFileLocation, "w") as json_model_file:
  json_model_file.write(model.to_json())
  model.save_weights(SavedWeightsFileLocation)

print('Test accuracy:', score[ 1 ])
print("Validation accuracy - %s" % score_validation[ 1 ])
print("Test accuracy - %s" % score[ 1 ])

pyplot.plot(history.history[ 'acc' ])
pyplot.plot(history.history[ 'val_acc' ])
pyplot.title('Model accuracy')
pyplot.ylabel('Accuracy')
pyplot.xlabel('Epoch')
pyplot.legend([ 'train', 'validation' ], loc = 'upper left')
pyplot.show()

pyplot.plot(history.history[ 'loss' ])
pyplot.plot(history.history[ 'val_loss' ])
pyplot.title('Model loss')
pyplot.ylabel('Loss')
pyplot.xlabel('Epoch')
pyplot.legend(['train', 'validation'], loc = 'upper left')
pyplot.show()

del soundwaves
del genres
gc.collect()