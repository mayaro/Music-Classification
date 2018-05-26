from pydub import AudioSegment
import multiprocessing
import threading
import os
import pathlib

from src.TrainingAudioData import crop_audio_file

def worker_processing( Id, Jobs):
  while not Jobs.empty():
    job = Jobs.get()

    print('Worker %s got job %s' % (Id, job))
    
    featured_interval = crop_audio_file(job)

    pathlib.Path('./learning_data/' + job.split('/')[-2]).mkdir(parents=True, exist_ok=True)
    with open('./learning_data/' + '/'.join(job.split('/')[-2:]), 'wb') as f:
      featured_interval.export(f, parameters=['-q:a', '90'])

    print('Worker %s finished job %s' % (Id, job))

Jobs = multiprocessing.Queue()
workers = []

for _, genres, _ in os.walk('X:/Facultate/MusicClassification/raw_learning_data'):
  for genre in genres:
    for current_path, _, songs in os.walk('X:/Facultate/MusicClassification/raw_learning_data/' + genre):
      for song_name in songs:
        full_file_name = current_path + '/' + song_name
        
        Jobs.put(full_file_name)

number_of_cpus = multiprocessing.cpu_count()
for i in range(number_of_cpus - 1):
  print('Starting worker %s' % i)

  p = threading.Thread( target=worker_processing, args=(i, Jobs) )
  workers.append(p)

  p.start()

for p in workers:
  p.join()