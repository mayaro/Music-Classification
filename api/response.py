class Response(object):
  
  error = None

  genres = []

  filename = ''

  def __init__(self, error, genres, filename):
    self.error = error
    self.genres = genres
    self.filename = filename

  def serialize(self):
    return {
      'error': self.error,
      'genres': self.genres,
      'filename': self.filename
    }