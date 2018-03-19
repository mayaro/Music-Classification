"""Utils.py

  Module that handles simple operations that are used in multiple places.

"""

import os

class CLIOperationsHandler(object):
  """Class used for handling CLI operations

    Args:
      cli_arguments ( class 'list' ): The cli-arguments passed as received from sys.argv.

    Attributes:
      cli_arguments ( class 'list' ): The cli-arguments passed as received from sys.argv.

  """

  def __init__(self, cli_arguments):

    self.cli_arguments = cli_arguments

  """Ensure the reloading of the .npy files if it was specified in the cli arguments.
    
    If the "--reload-npy" was not passed as a command argument the npy files will not be recreated.
    
    Args:
      directory ( string ): The directory where the previous npy files were located.
  
  """
  def handle_reload_npy( self, directory ):

    if self.__should_reload_npy( ):
      self.__remove_npy(directory)

  """Specify whether the script should remove the already created .npy files.
    
    Useful when the input files changed, as to recreate feature representation for them.

    Usage:
      python train.py --remove-npy

  """
  def __should_reload_npy( self ):
    return self.cli_arguments.find('--remove-npy') != -1

  """Remove the .npy files from the given directory.
    
    Args:
      directory ( string ): The directory that contains the .npy files.
  
  """
  def __remove_npy( self, directory ):
    for file in os.listdir(directory):
      if file.endswith('.npy'):
        os.remove(os.path.join(directory, file))
