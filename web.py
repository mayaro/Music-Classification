from __future__ import unicode_literals, division
from flask import Flask, request, jsonify
from api.response import Response
from configparser import ConfigParser

from urllib.request import Request, urlopen
from pydub import AudioSegment
from io import BytesIO
from pytube import YouTube
import pafy
import sys
import os
from random import randint

app = Flask(__name__)

config = ConfigParser()
config.read('config.ini')

base_path = config.get('Models', 'BaseDirectory')
model_path = base_path + config.get('Models', 'ModelPath')
weights_path = base_path + config.get('Models', 'WeightsPath')

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
    # video = pafy.new(url_param, ydl_opts={'prefer_insecure': False, 'http_chunk_size': '0.1M'})
    # bestaudio = video.getbestaudio()
    # filename = bestaudio.d  ownload(filepath='C:/Users/murar/AppData/Local/music_classification')

    yt = YouTube(url_param)
    yt.register_on_progress_callback(print_progress)
    stream = yt.streams.filter(only_audio=True, adaptive=True).order_by('bitrate').last()
    filename = 'C:/Users/murar/AppData/Local/music_classification/' + normalize_filename(stream.default_filename)
    stream.download(
      output_path='C:/Users/murar/AppData/Local/music_classification/',
      filename=''.join(normalize_filename(stream.default_filename).split('.')[:-1])
    )

  except Exception as e:
    resp = Response('{0}'.format(e), None).serialize()
    return jsonify(resp), 500
  
  try:
    audio = AudioSegment.from_file(filename)
    duration = len(audio)
    featured_interval = audio[duration / 2 - 10000 : duration / 2 + 10000]
  except Exception as e:
    resp = Response('{0}'.format(e), None).serialize()
    os.remove(filename)
    return jsonify(resp), 500

  os.remove(filename)
  return jsonify(Response(None, [len(featured_interval)]).serialize())

if __name__ == '__main__':
  app.run(debug=True, host='0.0.0.0', port=80)