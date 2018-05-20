"""TrainingAudioData.py

  This module contains the functionality for handling training data files.

"""

import os
import librosa
import random
import numpy
import multiprocessing
import keras
import threading

AudioOffset = 0.0
AudioDurationLimit = 20.0
SongSamples = 1320000
FFTWindowSize = 2048
HopLength = 128

class Job:
  def __init__( self, filename, genre ):
    self.filename = filename
    self.genre = genre

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
      'populara': 1,
      'reggae': 2,
      'classical': 3,
      'folk': 4
    }

  """Function used to get the .npy models for the training datafiles.

    Returns:
      :obj: Object containing mel samples for each audio file found in the training data directory
  
  """
  def get_mel_samples( self ):
    music_data = []
    genre_data = []

    Jobs = multiprocessing.Queue()
    workers = []

    temp_files = []
    genres = []

    for genre, _ in self.genres.items():
      for _, _, files in os.walk(self.directory + genre):
        for file_name in files:
          full_file_name = self.directory + genre + '/' + file_name

          temp_files.append(full_file_name)
          genres.append(self.genres[genre])

    temp = list(zip(temp_files, genres))
    random.shuffle(temp)
    temp_files, genres = zip(*temp)

    for idx, val in enumerate(temp_files):
      Jobs.put(
        Job(val, genres[idx])
      )

    number_of_cpus = multiprocessing.cpu_count()
    for i in range(number_of_cpus - 1):
      print('Starting worker %s' % i)

      p = threading.Thread( target=TrainingAudioData.worker_processing, args=(i, Jobs, music_data, genre_data) )
      workers.append(p)

      p.start()

    for p in workers:
      p.join()

    song_data_array = numpy.array(music_data)
    genre_data_array = keras.utils.to_categorical(genre_data, len(self.genres))

    return song_data_array, genre_data_array


  @staticmethod
  def process_audio_file( file_path ):
    signal, sampling_rate = librosa.load(file_path,
                                         offset = AudioOffset,
                                         duration = AudioDurationLimit)

    if signal.size == 0:
      print('File ' + file_path + ' could not be loaded.')
      return None

    feature = librosa.feature.melspectrogram(signal[ :SongSamples ], sr = sampling_rate,
                                             n_fft = FFTWindowSize, hop_length = HopLength).T[:3328, ]

    return feature

  @staticmethod
  def worker_processing( Id, Jobs, music_data, genre_data ):
    while not Jobs.empty():
      job = Jobs.get()

      print('Worker %s got job %s' % (Id, job.filename))
      features = TrainingAudioData.process_audio_file(job.filename)

      music_data.append(features)
      genre_data.append(job.genre)

      print('Worker %s finished job %s' % (Id, job.filename))      