from predictor import Predictor
import time
import numpy as np
import datetime 
import os
import soundfile as sf
from io import BytesIO
from threading import Thread
import shutil
from queue import Queue
import secrets
from synthesizer import Synthesizer

predictor = Predictor()
synthesizer = Synthesizer()

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
        self.pred_que = Queue()
        self.data_que = Queue()
        data_dir = f'{root_path}/data/'
        self.data_dirs = {
            'unknown' : f'{data_dir}/unknown',
            'adele' : f'{data_dir}/adele',
            'hilfe' : f'{data_dir}/hilfe',
        }
        for i in self.data_dirs.values(): os.makedirs(i,exist_ok=True)

    def save_audio(self,y,label):
        dst_pth = os.path.join(self.dest_folder,label)
        os.makedirs(dst_pth,exist_ok=True)
        file_name = f'{secrets.token_hex(5)}_{label}.wav'
        file_path = os.path.join(dst_pth,file_name)
        sf.write(file_path,y,predictor.target_sr)
        tmp_path = file_path.replace('\\','/').split('static')[-1]
        p = f'/static{tmp_path}'
        return p

    def _proceed_prediction(self):
        try:
            if not self.pred_que.empty():
                bytes_io = self.pred_que.get()
                y, (result,score), sr = predictor.predict(bytes_io,self.sr)
                if result is not None:
                    s_result = 'unknown'
                    if result != 'unknown':
                        s_result = synthesizer.synthesize(y,sr) or 'Unknown'
                    audio_path = self.save_audio(y,result)
                    self.results.append({
                        'prediction':result, 
                        's_prediction':s_result, 
                        'score':score,
                        # 'score':f'{score}%',
                        'path': audio_path,
                        'word_model': 'adele'
                    })
        except Exception as e: print(e)

    def predict(self):
        try:
            if not self.data_que.empty():
                data = self.data_que.get()
                b = BytesIO()
                sf.write(b,data,self.sr,format='WAV')
                b.seek(0)  
                self.pred_que.put(b)
                self._proceed_prediction()
                # Thread(target=self._proceed_prediction).start()
        except Exception as e: 
            print(e)

    def process(self):
        counter = 0
        while True:
            if self.audio_data.size > self.sr: 
                # self.data = self.audio_data[:self.sr]
                # self.audio_data = self.audio_data[self.sr:]

                self.data[:self._l] = self.data[self._l:]
                self.data[self._l:] = self.audio_data[:self._l]
                self.audio_data = self.audio_data[self._l:]

                self.data_que.put(self.data.copy())
                self.predict()
            if counter == 10000:
                time.sleep(5)
                counter = 0
            counter += 1
    
    def start_process(self):
        Thread(target=self.process).start()

    def reset(self):
        self.audio_data = np.array([])
        self.data = np.zeros(self.sr)
        shutil.rmtree(self.dest_folder)

    def move(self,path,label):
        try:
            shutil.move(path,self.data_dirs[label])
        except: pass
