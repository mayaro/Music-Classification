from json import dumps as to_json
from numpy import round

class Response(object):
  
  error = None

  genres = []

  filename = ''

  def __init__(self, error, genres, filename):
    self.error = error

    self.genres = to_json(round(genres, decimals=2).tolist(), separators=(',', ':')) if genres is not None else []

    self.filename = filename

  def serialize(self):
    return {
      'error': self.error,
      'genres': self.genres,
      'filename': self.filename
    }