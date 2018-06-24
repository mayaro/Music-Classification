import os
import numpy
from src.TrainingData import TrainingData

LEARNING_DATASET_LOCATION = os.getcwd() + '/learning_data/'
LEARNING_MODELS_LOCATION = os.getcwd() + '/models/'
SONGS_MODEL_FILE = LEARNING_MODELS_LOCATION + 'songs_data.npy'
GENRES_MODEL_FILE = LEARNING_MODELS_LOCATION + 'genres_data.npy'

songs, genres = TrainingData(LEARNING_DATASET_LOCATION)\
  .get_mel_samples()

numpy.save(SONGS_MODEL_FILE, songs)
numpy.save(GENRES_MODEL_FILE, genres)