from __future__ import unicode_literals
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from api.response import Response
from configparser import ConfigParser

from pydub import AudioSegment
from pytube import YouTube
from pathlib import Path

from src.Utils import remove_extension_from_filename

import os
import json

app = Flask(__name__)
cors = CORS(app)

app.config['CORS-HEADERS'] = 'Content-Type'

model = None

config = ConfigParser()
config.read('config.ini')

base_path = config.get('Models', 'BaseDirectory')
model_path = base_path + config.get('Models', 'ModelPath')
weights_path = base_path + config.get('Models', 'WeightsPath')
genre_names = config.get('Models', 'Genres').split(',')

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
@cross_origin()
def index():
  url_param = request.args.get('url')
  
  if url_param is None:
    resp = Response('Missing URL param', None).serialize()
    return jsonify(resp), 400
  
  filepath = ''
  featured_interval = None

  try: 
    yt = YouTube(url_param)
    yt.register_on_progress_callback(print_progress)
    stream = yt.streams.filter(only_audio=True, adaptive=True).order_by('bitrate').last()
    
    normalized_filename = normalize_filename(stream.default_filename)
    filepath = TempOutputDir + normalized_filename

    if Path(filepath).is_file():
      print('File {} already there'.format(filepath))
    else:
      stream.download(
        output_path=TempOutputDir,
        filename=remove_extension_from_filename(normalized_filename)
      )
      print('\n')

      audio = AudioSegment.from_file(filepath)
      # 3-second offset
      duration = len(audio) - 6000
      # Leave 500 ms on each end
      hop = int((duration - 1000) / 41)

      audio = audio[500 : duration - 500]
      # temp = audio[hop - 500 : hop + 500]

      samples = AudioSegment.empty()
      for i in range(1, 41):
        newSample = audio[hop * i - 250 : hop * i + 250]
        samples += newSample

      os.remove(filepath)
      with open(filepath, 'wb') as f:
        samples.export(f, parameters=['-q:a', '90'])

    song_samples = process_audio_file(filepath)
    formatted_song_samples = np.stack(np.split(song_samples, 26, axis=0), axis=0)

    global graph
    with graph.as_default():
      prediction = model.predict(formatted_song_samples, verbose = 1)

    score = np.around(np.mean(prediction, axis=0), decimals=4)
  except Exception as e:
    resp = Response('{0}'.format(e), None, filepath).serialize()
    return jsonify(resp), 500  

  response = Response(
    None,
    { genre_names[i]: '%.2f' % score[i] for i in range(len(genre_names)) },
    remove_extension_from_filename(normalized_filename)
  ).serialize()
  return jsonify(response)


"""
  The following code will only be run on the main thread
  as otherwise Keras will try to load the model each time this module is accessed.

  Also, tf.get_default_graph is called because of incompatibilities between Keras's Tensorflow backend
  and the asyncrony that webservers use.

  Also, Flask cannot run the web server in debug as it runs the web.py script twice
  and this makes tensorflow fail as it is trying to create 2 CUDA devices.
"""
if __name__ == '__main__':
  model = load_model()
  graph = tf.get_default_graph()

  app.run(host='0.0.0.0', port=80)