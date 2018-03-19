"""train.py

  This module contains the necessary functionality for training a neural network
    using the audio dataset location.

  Todo:
    * Get the mel-frequency samples
    * Save .npy models for not needing to create them each time the neural net is started.

"""

LEARNING_DATASET_LOCATION = 'C:/Users/murar/MusicClassification/learning_data/'
SONGS_MODEL_FILE = LEARNING_DATASET_LOCATION + 'songs_data.npy'
GENRES_MODEL_FILE = LEARNING_DATASET_LOCATION + 'genres_data.npy'

import os
import sys
import numpy
from src.TrainingAudioData import TrainingAudioData
from src.Utils import CLIOperationsHandler

cli_arguments = str(sys.argv)
cli_operations_handler = CLIOperationsHandler(cli_arguments)

cli_operations_handler.handle_reload_npy(LEARNING_DATASET_LOCATION)

if os.path.isfile(SONGS_MODEL_FILE) and os.path.isfile(GENRES_MODEL_FILE):

  songs_data = numpy.load(SONGS_MODEL_FILE)
  genres_data = numpy.load(GENRES_MODEL_FILE)

else:

  songs_data, genres_data = TrainingAudioData(LEARNING_DATASET_LOCATION)\
    .get_mel_samples()

  numpy.save(SONGS_MODEL_FILE, songs_data)
  numpy.save(GENRES_MODEL_FILE, genres_data)