"""train.py

  This module contains the necessary functionality for testing a neural network's model against a given song.

  @description
    - Will load the previously trained model from a path,
    - Compile the model
    - Test it with a given file and print out the results

"""

from sys import argv

DefaultModelDirectory = 'C:/Users/murar/MusicClassification/models/'
ModelPath = DefaultModelDirectory + 'model.json'
WeightsPath = DefaultModelDirectory + 'weights.h5'

# Get the path of the file that needs to be classified
arguments = argv[1:]
if len(arguments) <= 0:
  print('Please provide the file path for the file that you want classified')
  exit(1)

song_path = arguments[0]

# Only import keras if the script has been run correctly
import keras
import numpy as np
from keras.models import model_from_json
from src.TrainingAudioData import TrainingAudioData

# Load the model from a JSON file
# It should have been saved using model.to_json() for this to work
with open(ModelPath, "r") as json_file:
  model = model_from_json(json_file.read())
  model.load_weights(WeightsPath)
  
  print('Model loaded!')

# Stochastic Gradient Descent optimizer used for compiling the keras model
sgd = keras.optimizers.SGD(lr = 0.001, momentum = 0.9, decay = 1e-5, nesterov = True)

# Compile the model for using it to classify the song
model.compile(loss = keras.losses.categorical_crossentropy,
              optimizer = sgd,
              metrics = [ 'accuracy' ])

# Compute and format the soundwave samples used in the classification
try:
  song_samples = TrainingAudioData.process_audio_file(song_path)
  formatted_song_samples = np.stack(np.split(song_samples, 5, axis=0), axis=0)
except FileNotFoundError:
  print('Given file path was not found, exiting...')
  exit(2)

# Compute a prediction using the previously compiled Keras model and format it
prediction = model.predict(formatted_song_samples, verbose = 1)
score = np.around(np.mean(prediction, axis=0), decimals=1)

print(score)