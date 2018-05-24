class Response(object):
  
  error = None

  genres = []

  def __init__(self, error, genres):
    self.error = error
    self.genres = genres

  def serialize(self):
    return {
      'error': self.error,
      'genres': self.genres
    }