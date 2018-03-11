import os

# @Class: AudioData
# @Description
# Class used for creating and getting the Mel samples that are to be used in training from a set of data
class AudioData( object ):

  def __init__( self, directory ):
    self.directory = directory
    self.genres = {
      'metal': 1,
      'country': 2,
      'classical': 3,
      'reggae': 4
    }

  def get_mel_samples( self ):
    for genre, _ in self.genres.items():
      for r, s, files in os.walk(self.directory + genre):
        for file in files:
          print(file)