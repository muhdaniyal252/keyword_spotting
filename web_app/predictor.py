
import librosa
from transformers import pipeline

class Predictor:

    def __init__(self):
        model_pth = 'D:/model_code/server_models/wav2vec2/trial_2/wav2vec2-finetune'
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
        
