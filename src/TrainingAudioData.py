"""TrainingAudioData.py

  This module contains the functionality for handling training data files.

"""

import os
import librosa
import numpy
import keras

AUDIO_OFFSET = 8.0
AUDIO_LIMIT_DURATION = 60.0
AUDIO_SONG_SAMPLES = 660000
FFT_WINDOW_SIZE = 2048
HOP_LENGTH = 512

class TrainingAudioData(object):
  """Class used for creating and handling audio data

    Args:
      directory ( string ): The relative path for the training data files.

    Attributes:
      directory ( string ): The relative path for the training data files.
      genres ( :obj ): The genres used at training.
        Only files from these genres will be taken into account when extracting the training data.
        These must match the names of the directories in the training dataset.

  """

  def __init__( self, directory ):
    self.directory = (directory, directory + '/')[not directory.endswith('/')]
    self.genres = {
      'metal': 0,
      'country': 1
    }

  """Function used to get the .npy models for the training datafiles.

    Returns:
      :obj: Object containing mel samples for each audio file found in the training data directory
  
  """
  def get_mel_samples( self ):
    music_data = []
    genre_data = []

    for genre, _ in self.genres.items():
      index = 0
      for _, _, files in os.walk(self.directory + genre):
        for file_name in files:
          full_file_name = self.directory + genre + '/' + file_name

          feature = TrainingAudioData.__process_audio_file(full_file_name)

          music_data.append(feature)
          genre_data.append(self.genres[genre])

          print('Parsed audio file \'' + full_file_name + '\'.')

          index += 1

          if index == 5:
            index = 0
            break

    song_data_array = numpy.array(music_data)
    genre_data_array = keras.utils.to_categorical(genre_data, len(self.genres))

    return song_data_array, genre_data_array


  @staticmethod
  def __process_audio_file( file_path ):
    signal, sampling_rate = librosa.load(file_path,
                                         offset = AUDIO_OFFSET,
                                         duration = AUDIO_LIMIT_DURATION)

    if signal.size == 0:
      print('File ' + file_path + ' could not be loaded.')
      return None

    feature = librosa.feature.melspectrogram(signal[ :AUDIO_SONG_SAMPLES ], sr = sampling_rate,
                                             n_fft = FFT_WINDOW_SIZE, hop_length = HOP_LENGTH).T[:1280, ]

    return feature