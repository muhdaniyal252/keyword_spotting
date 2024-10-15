import sounddevice as sd
from scipy.io.wavfile import write
import os

path = 'D:/data_code/data/neg_data/conversation/br_de/1s_chunks/'
os.makedirs(path,exist_ok=True)
fs = 44100
buffer_duration = 4.5 # seconds
buffer_size = int(buffer_duration * fs)

for i in range(2):
    buffer = sd.rec(buffer_size, channels=1, dtype='float32')
    sd.wait()
    buffer = buffer.squeeze()[int(fs * 2):]
    write(f'{path}brde_{i}.wav',fs,buffer)