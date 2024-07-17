from predictor import Predictor
from threading import Thread
import time
import numpy as np
import datetime 
import os
import soundfile as sf
from io import BytesIO
from threading import Thread
import shutil
from queue import Queue

predictor = Predictor()

class Handler:

    def __init__(self,root_path):
        current_datetime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
        self.dest_folder = os.path.join(root_path,'static','audios',current_datetime)
        os.makedirs(self.dest_folder,exist_ok=True)
        self.audio_data = np.array([])
        self.sr = 0
        self._l = 0
        self.data = np.zeros(0)
        self.results = list()
        self.q = Queue()

    def save_audio(self,y,label):
        dst_pth = os.path.join(self.dest_folder,label)
        os.makedirs(dst_pth,exist_ok=True)
        file_name = f'{len(os.listdir(dst_pth))}_{label}.wav'
        file_path = os.path.join(dst_pth,file_name)
        sf.write(file_path,y,predictor.target_sr)
        tmp_path = file_path.replace('\\','/').split('static')[-1]
        p = f'/static{tmp_path}'
        return p

    def _proceed_prediction(self):
        try:
            if not self.q.empty():
                bytes_io = self.q.get()
                y, (result,score) = predictor.predict(bytes_io,self.sr)
                if result is not None:
                    audio_path = self.save_audio(y,result)
                    self.results.append({
                        'prediction':result, 
                        'score':f'{score}%',
                        'path': audio_path
                    })
        except Exception as e: print(e)

    def predict(self):
        try:
            b = BytesIO()
            sf.write(b,self.data,self.sr,format='WAV')
            b.seek(0)  
            self.q.put(b)
            # self._proceed_prediction()
            Thread(target=self._proceed_prediction).start()
        except Exception as e: 
            print(e)

    def process(self):
        while True:
            if self.audio_data.size > self.sr: 
                self.data[:self._l] = self.data[self._l:]
                self.data[self._l:] = self.audio_data[:self._l]
                self.audio_data = self.audio_data[self._l:]
                self.predict()
            time.sleep(0.1)
    
    def start_process(self):
        Thread(target=self.process).start()

    def reset(self):
        self.audio_data = np.array([])
        self.data = np.zeros(self.sr)
        shutil.rmtree(self.dest_folder)