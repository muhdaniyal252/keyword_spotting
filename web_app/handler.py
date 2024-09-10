from a_predictor import A_Predictor
from h_predictor import H_Predictor
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

a_predictor = A_Predictor()
h_predictor = H_Predictor()
synthesizer = Synthesizer()

class Handler:

    def __init__(self,root_path):
        current_datetime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
        self.dest_folder = os.path.join(root_path,'static','audios',current_datetime)
        os.makedirs(self.dest_folder,exist_ok=True)
        self.a_audio_data = np.array([])
        self.h_audio_data = np.array([])
        self.sr = 0
        self._al = 0
        self._hl = 0
        self.a_data = np.zeros(0)
        self.h_data = np.zeros(0)
        self.results = list()
        self.a_pred_que = Queue()
        self.h_pred_que = Queue()
        self.a_data_que = Queue()
        self.h_data_que = Queue()
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
        sf.write(file_path,y,a_predictor.target_sr)
        tmp_path = file_path.replace('\\','/').split('static')[-1]
        p = f'/static{tmp_path}'
        return p

    def _a_proceed_prediction(self):
        try:
            if not self.a_pred_que.empty():
                bytes_io = self.a_pred_que.get()
                y, result, result_l, sr = a_predictor.predict(bytes_io,self.sr)
                if result is not None:
                    s_result = 'unknown'
                    if result != 'unknown':
                        s_result = synthesizer.synthesize(y,sr) or 'Unknown'
                    audio_path = self.save_audio(y,result)
                    self.results.append({
                        'prediction':result, 
                        'l_prediction':result_l,
                        's_prediction':s_result, 
                        'score':'--',
                        # 'score':f'{score}%',
                        'path': audio_path,
                        'word_model': 'adele'
                    })
        except Exception as e: print(e)
        
    def _h_proceed_prediction(self):
        try:
            if not self.h_pred_que.empty():
                bytes_io = self.h_pred_que.get()
                y, result, result_l, sr = h_predictor.predict(bytes_io,self.sr)
                if result is not None:
                    s_result = 'unknown'
                    if result != 'unknown':
                        s_result = synthesizer.synthesize(y,sr) or 'Unknown'
                    audio_path = self.save_audio(y,result)
                    self.results.append({
                        'prediction':result, 
                        'l_prediction':result_l,
                        's_prediction':s_result, 
                        'score':'--',
                        # 'score':f'{score}%',
                        'path': audio_path,
                        'word_model': 'hilfe'
                    })
        except Exception as e: print(e)

    def predict(self):
        try:
            if not (self.a_data_que.empty() and self.h_data_que.empty()) :
                a_data = self.a_data_que.get()
                a_b = BytesIO()
                sf.write(a_b,a_data,self.sr,format='WAV')
                a_b.seek(0)  
                self.a_pred_que.put(a_b)
                h_data = self.a_data_que.get()
                h_b = BytesIO()
                sf.write(h_b,h_data,self.sr,format='WAV')
                h_b.seek(0)  
                self.h_pred_que.put(h_b)
                self._a_proceed_prediction()
                self._h_proceed_prediction()
                # Thread(target=self._proceed_prediction).start()
        except Exception as e: 
            print(e)

    def process(self):
        counter = 0
        while True:
            if self.a_audio_data.size > self.sr and self.h_audio_data.size > int(self.sr*2.5): 

                self.a_data[:self._al] = self.a_data[self._al:]
                self.a_data[self._al:] = self.a_audio_data[:self._al]
                self.a_audio_data = self.a_audio_data[self._al:]

                self.a_data_que.put(self.a_data.copy())

                self.h_data[:self._hl] = self.h_data[self._hl:]
                self.h_data[self._hl:] = self.h_audio_data[:self._hl]
                self.h_audio_data = self.h_audio_data[self._hl:]

                self.h_data_que.put(self.h_data.copy())

                self.predict()
            if counter == 10000:
                time.sleep(5)
                counter = 0
            counter += 1
    
    def start_process(self):
        Thread(target=self.process).start()

    def reset(self):
        self.a_audio_data = np.array([])
        self.h_audio_data = np.array([])
        self.a_data = np.zeros(self.sr)
        shutil.rmtree(self.dest_folder)

    def move(self,path,label):
        try:
            shutil.move(path,self.data_dirs[label])
        except: pass
