"""TrainingAudioData.py

  This module contains the functionality for handling training data files.

  Todo:
    * Create function that returns the mel samples for all training data files.

"""

import os

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
      'metal': 1,
      'country': 2,
      'classical': 3,
      'reggae': 4
    }

  """Function used to get the .npy models for the training datafiles.

    Returns:
      :obj: Object containing mel samples for each audio file found in the training data directory
  
  """
  def get_mel_samples( self ):
    for genre, _ in self.genres.items():
      for _, _, files in os.walk(self.directory + genre):
        for file_name in files:
          full_file_name = self.directory + genre + '/' + file_name
          print(full_file_name)
