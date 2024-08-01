
import librosa
# from transformers import pipeline
import tensorflow as tf
import numpy as np

class _Predictor:

    def __init__(self):
        model_pth = r"D:\model_code\server_models\wav2vec2\trail_1\wav2vec2-finetune"
        self.pipe = pipeline("audio-classification", model=model_pth)
        self.target_sr = 16000


    def get_label(self,x):
        lbl = None
        mx_n = -1
        for i in x:
            if i['score'] > mx_n:
                mx_n = i['score']
                lbl = i['label']
        # return lbl,round(mx_n*100,2)
        if lbl != 'unknown' and mx_n > 0.83:
            return lbl,round(mx_n*100,2)
        return 'unknown', round(mx_n*100,2)
        if mx_n > 0.99: return f'{lbl} - {mx_n}' if y != 'unknown' else None
        return None
    
    def predict(self,data,sr):
        
        data,_ = librosa.load(data,sr=sr)
        y = librosa.resample(data,orig_sr=sr,target_sr=self.target_sr)
        _result = self.pipe(y)
        print(_result)
        return y, self.get_label(_result)

class Predictor:
    
    def __init__(self):
        self.model = tf.keras.models.load_model(r"D:\model_code\server_models\mobile_net\trail_1\2\16k_1s_melspec-nfft-1024_a_h_cnn_dense_model.keras")
        self.target_sr = 16000
        self.max_seconds = 1
        self.label = {
            0:'unknown',1:'adele',2:'hilfe'
        }
        self.pad_or_trunc = lambda a,i : a[0:i] if len(a) > i else a if len(a) == i else np.pad(a,(0, (i-len(a))))

    def process_data(self,y,sr,max_seconds):
        y = self.pad_or_trunc(y,sr*max_seconds)
        features = librosa.feature.melspectrogram(y=y,sr=sr,n_fft=1024)
        return features

    def get_features(self,waveform, sr):
        features = self.process_data(waveform,sr,self.max_seconds)
        shape = features.shape
        features = features.reshape([1,*shape,1])
        return features
    
    def predict(self,data,sr):
        # data,_ = librosa.load(data,sr=sr)
        y = librosa.resample(data,orig_sr=sr,target_sr=self.target_sr)
        features = self.get_features(y,sr=self.target_sr)
        pred = self.model.predict(features)
        return y,(self.label.get(np.argmax(pred),'unknown'),round(np.max(pred)*100,2))