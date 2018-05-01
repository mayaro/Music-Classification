"""train.py

  This module contains the necessary functionality for training a neural network
    using the audio dataset location.

  Todo:
    * Get the mel-frequency samples
    * Save .npy models for not needing to create them each time the neural net is started.

"""

LearningDatasetLocation = 'C:/Users/murar/MusicClassification/models/'
SoundwavesDataFileLocation = LearningDatasetLocation + 'songs_data.npy'
GenresDataFileLocation = LearningDatasetLocation + 'genres_data.npy'

SavedModelFileLocation = LearningDatasetLocation + 'model.json'
SavedWeightsFileLocation = LearningDatasetLocation + 'weights.h5'

ExecTimes = 150
CNN_TYPE = '1D'
input_shape = (128, 128)
batch_size = 100
epochs = 100

import os
import gc
import numpy

import matplotlib.pyplot as pyplot
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import keras

from src.AudioModel import TrainingModels
from src.AudioUtils import AudioUtils

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
test_accuracy_voted = [ ]

for execPeriod in range(ExecTimes):
  # Split the soundwave dateset into training and test
  inputs_train, inputs_test, outputs_train, outputs_test = train_test_split(
    soundwaves, genres, test_size = 0.1, stratify = genres)

  # Split the training dataset into training and validation
  inputs_train, inputs_validation, outputs_train, outputs_validation = train_test_split(
    inputs_train, outputs_train, test_size = 1 / 6, stratify = outputs_train)

  # Split the train, test and validation datasets in sizes of 128x128
  inputs_validation, outputs_validation = AudioUtils( ).splitsongs_melspect(
    inputs_validation, outputs_validation, CNN_TYPE)
  inputs_test, outputs_test = AudioUtils( ).splitsongs_melspect(
    inputs_test, outputs_test, CNN_TYPE)
  inputs_train, outputs_train = AudioUtils( ).splitsongs_melspect(
    inputs_train, outputs_train, CNN_TYPE)

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
    monitor = 'val_loss',
    min_delta = 0,
    patience = 2,
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

  # Also use a majority voting system for computing accuracy
  predicted_values = model.predict(inputs_test)
  
  max_predicted_values = numpy.argmax(predicted_values, axis = 1)
  voted_truth, voted_prediction = AudioUtils().voting(numpy.argmax(outputs_test, axis = 1), max_predicted_values)
  
  accuracy_voted = accuracy_score(voted_truth, voted_prediction)

  # Save training metrics
  validation_accuracy.append(score_validation[ 1 ])
  test_accuracy.append(score[ 1 ])
  test_history.append(history)
  test_accuracy_voted.append(accuracy_voted)

  # Print metrics
  print('Test accuracy:', score[ 1 ])
  print('Test accuracy for Majority Voting System:', accuracy_voted)

  # Print the confusion matrix for Voting System
  cm = confusion_matrix(voted_truth, voted_prediction)
  print(cm)

  print('Exec period %s' % execPeriod)

  gc.collect()

# Print the statistics
print("Validation accuracy - %s" % numpy.mean(validation_accuracy))
print("Test accuracy - %s" % numpy.mean(test_accuracy))
print("Test accuracy MVS - %s" % numpy.mean(test_accuracy_voted))

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