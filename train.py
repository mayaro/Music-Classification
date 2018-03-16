"""train.py

  This module contains the necessary functionality for training a neural network
    using the audio dataset location.

  Todo:
    * Get the mel-frequency samples
    * Save .npy models for not needing to create them each time the neural net is started.

"""

from src.TrainingAudioData import TrainingAudioData

audio_data = TrainingAudioData('./learning_data/')\
  .get_mel_samples()