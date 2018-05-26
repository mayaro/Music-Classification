import numpy as np

# Should be equal to the number of samples
SplitSize = 26

'''Split the samples arrays that were passed as arguments into songs.

  Description:
    Given two arrays of samples from multiple songs, the function will return the two arrays, splitted into songs.

  Args:
    inputs ( :obj:`np.array` of :obj:`np.float64` )
    outputs ( :obj:`np.array` of :obj:`np.float64` )

  Returns:
    :obj:`np.array`, :obj:`np.array`

'''
def split_songs( inputs, outputs ):
  final_inputs = [ ]
  final_outputs = [ ]

  for index, song in enumerate(inputs):
    splitted_inputs = np.split(song, SplitSize)

    for input_soundwave in splitted_inputs:
      final_inputs.append(input_soundwave)
      final_outputs.append(outputs[ index ])

  return np.array(final_inputs), np.array(final_outputs)

"""Remove the extension from filenames.

  Args:
    filename ( string )

  Returns:
    string: The filename without extension
"""
def remove_extension_from_filename(filename):
  return '.'.join(filename.split('.')[:-1])