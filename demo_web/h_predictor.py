import tensorflow as tf
import numpy as np
import noisereduce as nr
from predictor import Predictor

class H_Predictor(Predictor): #custom model 1 - #Adele
    
    def __init__(self):
        super().__init__()
        self.lite_model = tf.lite.Interpreter(model_path="/shareddrive/working/model_code/models/custom_model_4/trail_2/16k_melspec-nfft-1024_h_cnn_dense_model.tflite")
        self.model = tf.keras.models.load_model('/shareddrive/working/model_code/models/custom_model_4/trail_2/16k_melspec-nfft-1024_h_cnn_dense_model.keras')
        self.lite_model.allocate_tensors()
        self.input_details = self.lite_model.get_input_details()
        self.output_details = self.lite_model.get_output_details()
        self.max_seconds = 2.5
        self.label = {
            0:'unknown',1:'Hilfe-Hilfe'
        }

class _H_Predictor: #dummy
    
    def __init__(self):
        self.target_sr = 16000
    
    def predict(self,data,sr):
        y = np.random.random(16000)
        return y,('unknown',89.3)
