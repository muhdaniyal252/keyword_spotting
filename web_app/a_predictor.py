
import librosa
import tensorflow as tf
import numpy as np
import noisereduce as nr


class A_Predictor: #custom model 1 - #Adele
    
    def __init__(self):
        self.lite_model = tf.lite.Interpreter(model_path="/shareddrive/working/model_code/models/custom_model_4/trail_2/16k_melspec-nfft-1024_h_cnn_dense_model.tflite")
        self.model = tf.keras.models.load_model('/shareddrive/working/model_code/models/custom_model_4/trail_2/16k_melspec-nfft-1024_h_cnn_dense_model.keras')
        self.lite_model.allocate_tensors()
        self.input_details = self.lite_model.get_input_details()
        self.output_details = self.lite_model.get_output_details()
        self.target_sr = 16000
        self.max_seconds = 1
        self.label = {
            0:'unknown',1:'adele'
        }
        self.pad_or_trunc = lambda a,i : a[0:i] if len(a) > i else a if len(a) == i else np.pad(a,(0, (i-len(a))))

    def process_data(self,y,sr,max_seconds):
        # y = nr.reduce_noise(y=y, sr=sr,n_fft=1024)
        # y[np.isnan(y)] = 0
        y = self.pad_or_trunc(y,sr*max_seconds)
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

class _A_Predictor: #dummy
    
    def __init__(self):
        self.target_sr = 16000
    
    def predict(self,data,sr):
        y = np.random.random(16000)
        return y,('unknown',89.3)
