import librosa

class Predictor:

    def __init__(self) -> None:
        self.target_sr = 16000
        self.max_seconds = -1
        self.lite_model = None
        self.model = None
        self.input_details = None
        self.output_details = None
        self.label = {}
        self.pad_or_trunc = lambda a,i : a[0:i] if len(a) > i else a if len(a) == i else np.pad(a,(0, (i-len(a))))

    def process_data(self,y,sr,max_seconds):
        # y = nr.reduce_noise(y=y, sr=sr,n_fft=1024)
        # y[np.isnan(y)] = 0
        y = self.pad_or_trunc(y,int(sr*max_seconds))
        features = librosa.feature.melspectrogram(y=y,sr=sr,n_fft=1024)
        return features

    def get_features(self,waveform, sr):
        features = self.process_data(waveform,sr,self.max_seconds)
        shape = features.shape
        features = features.reshape([1,*shape,1])
        return features
    
    def predict(self,data,sr):
        data,_ = librosa.load(data,sr=sr)
        y = librosa.resample(data,orig_sr=sr,target_sr=self.target_sr)
        features = self.get_features(y,sr=self.target_sr)
        
        pred = self.model.predict(features)

        self.lite_model.set_tensor(self.input_details[0]['index'], features)
        self.lite_model.invoke()
        output_data = self.lite_model.get_tensor(self.output_details[0]['index'])
        lite_pred = output_data

        return y, self.label.get(1 if pred[0] > 0.5 else 0, 'unknown'), self.label.get(1 if lite_pred[0] > 0.5 else 0, 'unknown'), self.target_sr
        # return y,(self.label.get(np.argmax(pred) if np.max(pred) >= 0.97 else 0,'unknown'),round(np.max(pred)*100,2))
