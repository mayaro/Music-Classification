"""train.py

  This module contains the necessary functionality for training a neural network
    using the audio dataset location.

  Todo:
    * Get the mel-frequency samples
    * Save .npy models for not needing to create them each time the neural net is started.

"""

LearningDatasetLocation = 'E:/FACULTATE/MusicClassification/models/'
SoundwavesDataFileLocation = LearningDatasetLocation + 'songs_data.npy'
GenresDataFileLocation = LearningDatasetLocation + 'genres_data.npy'

SavedModelFileLocation = LearningDatasetLocation + 'model.json'
SavedWeightsFileLocation = LearningDatasetLocation + 'weights.h5'

ExecTimes = 100
input_shape = (128, 128)
batch_size = 100
epochs = 100

import os
import gc
import numpy
import keras

import matplotlib.pyplot as pyplot
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from src.AudioModel import TrainingModels
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

print('Soundwaves format:\t' + str(soundwaves.shape))
print('Genres format:\t' + str(genres.shape))

# Some training metrics
validation_accuracy = [ ]
test_history = [ ]
test_accuracy = [ ]

for execPeriod in range(ExecTimes):
  # Split the soundwave dateset into training and test
  inputs_train, inputs_test, outputs_train, outputs_test = train_test_split(
    soundwaves, genres, test_size = 0.1, stratify = genres)

  # Split the training dataset into training and validation
  inputs_train, inputs_validation, outputs_train, outputs_validation = train_test_split(
    inputs_train, outputs_train, test_size = 1 / 6, stratify = outputs_train)

  # Split the train, test and validation datasets in sizes of 128x128
  inputs_validation, outputs_validation = split_songs(
    inputs_validation, outputs_validation)
  inputs_test, outputs_test = split_songs(
    inputs_test, outputs_test)
  inputs_train, outputs_train = split_songs(
    inputs_train, outputs_train)

  model = TrainingModels.CNNMelspec(input_shape)

  print("Training dataset shape: {0}".format(inputs_train.shape))
  print("Validation dataset shape: {0}".format(inputs_validation.shape))
  print("Test dataset shape: {0}\n".format(inputs_test.shape))
  print("CNN size: %s\n" % model.count_params( ))

  # optimizer used for compiling the model
  sgd = keras.optimizers.SGD(lr = 0.001, momentum = 0.9, decay = 1e-5, nesterov = True)

  # Compile the model using the SGD optimizer
  model.compile(
    loss = keras.losses.categorical_crossentropy,
    optimizer = sgd,
    metrics = [ 'accuracy' ]
  )

  # Callback used to stop epoch training if the loss is small enough
  stop_callback = keras.callbacks.EarlyStopping(
    monitor = 'val_acc',
    min_delta = 0,
    patience = 3,
    verbose = 0,
    mode = 'auto'
  )

  # Fit the model
  history = model.fit(
    inputs_train, outputs_train,
    batch_size = batch_size,
    epochs = epochs,
    verbose = 1,
    validation_data = ( inputs_validation, outputs_validation ),
    callbacks = [ stop_callback ])

  # Start training and then validate the given number of epochs in one execution period
  score = model.evaluate(
    inputs_test, outputs_test,
    verbose = 0
  )
  score_validation = model.evaluate(
    inputs_validation, outputs_validation,
    verbose = 0
  )

  # Save training metrics
  validation_accuracy.append(score_validation[ 1 ])
  test_accuracy.append(score[ 1 ])
  test_history.append(history)

  # Print metrics
  print('Test accuracy:', score[ 1 ])
  print('Exec period %s' % execPeriod)

  gc.collect()

# Print the statistics
print("Validation accuracy - %s" % numpy.mean(validation_accuracy))
print("Test accuracy - %s" % numpy.mean(test_accuracy))

# Plot accuracy history
pyplot.plot(history.history[ 'acc' ])
pyplot.plot(history.history[ 'val_acc' ])
pyplot.title('Model accuracy')
pyplot.ylabel('Accuracy')
pyplot.xlabel('Epoch')
pyplot.legend([ 'train', 'validation' ], loc = 'upper left')
pyplot.show()

# Plot loss history
pyplot.plot(history.history[ 'loss' ])
pyplot.plot(history.history[ 'val_loss' ])
pyplot.title('Model loss')
pyplot.ylabel('Loss')
pyplot.xlabel('Epoch')
pyplot.legend(['train', 'validation'], loc = 'upper left')
pyplot.show()

# Save the model
with open(SavedModelFileLocation, "w") as json_model_file:
  json_model_file.write(model.to_json())
  model.save_weights(SavedWeightsFileLocation)

# Free memory at the end of the training
del soundwaves
del genres
gc.collect()