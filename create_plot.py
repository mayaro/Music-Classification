import librosa, librosa.display
import numpy, scipy, matplotlib.pyplot as plt, IPython.display as ipd

x_c, y_c = librosa.load('C:/Users/murar/AppData/Local/music_classification/Beethoven-Fur Elise.webm', duration=30)
x_m, y_m = librosa.load('C:/Users/murar/AppData/Local/music_classification/Metallica - Enter Sandman [Official Music Video].webm', duration=30)
x_r, y_r = librosa.load('C:/Users/murar/AppData/Local/music_classification/Bob Marley - Buffalo soldier.webm', duration=30)

plt.figure()

# SOUNDWAVES

# plt.subplot(311)
# plt.title('a')
# librosa.display.waveplot(x_c, y_c)
# plt.xlabel('')

# plt.subplot(312)
# plt.title('b')
# librosa.display.waveplot(x_m, y_m)
# plt.xlabel('')
# plt.ylabel('Amplitudinea piesei muzicale')

# plt.subplot(313)
# plt.title('c')
# librosa.display.waveplot(x_r, y_r)
# plt.xlabel('Intervalul de timp (s)')

# SPECTROGRAMS

spec = librosa.feature.melspectrogram(
  y=x_c,
  sr=y_c
)
plt.subplot(311)
plt.title('a')
librosa.display.specshow(
  librosa.power_to_db(
    spec,                                              
    ref=numpy.max
  ),
  y_axis='mel',
  fmax=8000,
  # x_axis='time'
)
plt.colorbar(format='%+2.0f dB')

spec = librosa.feature.melspectrogram(
  y=x_m,
  sr=y_m
)
plt.subplot(312)
plt.title('b')
librosa.display.specshow(
  librosa.power_to_db(
    spec,                                              
    ref=numpy.max
  ),
  y_axis='mel',
  fmax=8000,
  # x_axis='time'
)
plt.colorbar(format='%+2.0f dB')

spec = librosa.feature.melspectrogram(
  y=x_r,
  sr=y_r
)
plt.subplot(313)
plt.title('c')
librosa.display.specshow(
  librosa.power_to_db(
    spec,                                              
    ref=numpy.max
  ),
  y_axis='mel',
  fmax=8000,
  x_axis='time'
)
plt.colorbar(format='%+2.0f dB')


plt.show()