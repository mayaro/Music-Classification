from __future__ import unicode_literals, division
from flask import Flask, request, jsonify
from api.response import Response
from configparser import ConfigParser

from pydub import AudioSegment
from pytube import YouTube
from pathlib import Path
import os
import json

app = Flask(__name__)
model = None

config = ConfigParser()
config.read('config.ini')

base_path = config.get('Models', 'BaseDirectory')
model_path = base_path + config.get('Models', 'ModelPath')
weights_path = base_path + config.get('Models', 'WeightsPath')

def load_model():
  from tensorflow.python.keras.models import model_from_json
  from tensorflow.python.keras.optimizers import SGD
  from tensorflow.python.keras.losses import categorical_crossentropy
  
  with open(model_path, "r") as json_file:
    
    model = model_from_json(json_file.read())
    model.load_weights(weights_path)
    model._make_predict_function()

    sgd = SGD(lr = 0.001, momentum = 0.9, decay = 1e-5, nesterov = True)

    # Compile the model for using it to classify the song
    model.compile(loss = categorical_crossentropy,
                  optimizer = sgd,
                  metrics = [ 'accuracy' ])


    print('Model loaded!')

  return model

TempOutputDir = 'C:/Users/murar/AppData/Local/music_classification/'

import tensorflow as tf
import numpy as np
import librosa

AudioOffset = 0.0
AudioDurationLimit = 20.0
SongSamples = 1320000
FFTWindowSize = 2048
HopLength = 128



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

def normalize_filename(filename):
  return filename.replace('  ', ' ').strip()

def print_progress(stream, chunk, filehandle, bytes):
  streamsize = stream.filesize
  print('Downloading: {0:.0f}% - {1}'.format(
    (streamsize - bytes) / streamsize * 100,
    normalize_filename(stream.default_filename)
  ), end='\r', flush=True)

@app.route('/classify')
def index():
  url_param = request.args.get('url')
  
  if url_param is None:
    resp = Response('Missing URL param', None).serialize()
    return jsonify(resp), 400
  
  filename = ''
  featured_interval = None

  try: 
    yt = YouTube(url_param)
    yt.register_on_progress_callback(print_progress)
    stream = yt.streams.filter(only_audio=True, adaptive=True).order_by('bitrate').last()
    filename = TempOutputDir + normalize_filename(stream.default_filename)
    
    if Path(filename).is_file():
      print('File {} already there'.format(filename))
    else:
      stream.download(
        output_path=TempOutputDir,
        filename=''.join(normalize_filename(stream.default_filename).split('.')[:-1])
      )

      audio = AudioSegment.from_file(filename)
      duration = len(audio)
      featured_interval = audio[duration / 2 - 10000 : duration / 2 + 10000]
      
      os.remove(filename)
      with open(filename, 'wb') as f:
        featured_interval.export(f, parameters=['-q:a', '90'])

    song_samples = process_audio_file(filename)
    formatted_song_samples = np.stack(np.split(song_samples, 26, axis=0), axis=0)

    global graph
    with graph.as_default():
      prediction = model.predict(formatted_song_samples, verbose = 1)

    score = np.around(np.mean(prediction, axis=0), decimals=1)
  except Exception as e:
    resp = Response('{0}'.format(e), None, filename).serialize()
    return jsonify(resp), 500  

  return jsonify(Response(None, score, '.'.join(filename.split('.')[:-1])).serialize())


if __name__ == '__main__':
  model = load_model()

  graph = tf.get_default_graph()

  app.run(host='0.0.0.0', port=80)