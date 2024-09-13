import sounddevice as sd
from scipy.io.wavfile import write
import os

path = 'D:/data_code/data/neg_data/conversation/inforadio/1s_chunks/'
os.makedirs(path,exist_ok=True)
fs = 44100
buffer_duration = 2 # seconds
buffer_size = buffer_duration * fs

for i in range(3600):
    buffer = sd.rec(buffer_size, channels=1, dtype='float32')
    sd.wait()
    buffer = buffer.squeeze()[fs:]
    write(f'{path}info_{i}.wav',fs,buffer)