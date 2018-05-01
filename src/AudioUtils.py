import random
import librosa
import numpy as np
from scipy.stats import mode
from keras import backend as K

class AudioUtils(object):

  def __init__( self ):
    self.augment_factor = 5

  def splitsongs_melspect( self, X, y, cnn_type = '1D' ):
    temp_X = [ ]
    temp_y = [ ]

    for i, song in enumerate(X):
      song_slipted = np.split(song, self.augment_factor)
      for s in song_slipted:
        temp_X.append(s)
        temp_y.append(y[ i ])

    temp_X = np.array(temp_X)
    temp_y = np.array(temp_y)

    return temp_X, temp_y

  def voting( self, y_true, pred ):
    if y_true.shape[ 0 ] != pred.shape[ 0 ]:
      raise ValueError('Both arrays should have the same size!')

    # split the arrays in songs
    arr_size = y_true.shape[ 0 ]
    pred = np.split(pred, arr_size / self.augment_factor)
    y_true = np.split(y_true, arr_size / self.augment_factor)

    # Empty answers
    voting_truth = [ ]
    voting_ans = [ ]

    for x, y in zip(y_true, pred):
      voting_truth.append(mode(x)[ 0 ][ 0 ])
      voting_ans.append(mode(y)[ 0 ][ 0 ])

    return np.array(voting_truth), np.array(voting_ans)