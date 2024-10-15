import tensorflow as tf
import numpy as np
from predictor import Predictor

lite_model = r'D:\keyword_spotting\server_models\custom_model_4\trail_1\16k_melspec-nfft-1024_a_cnn_dense_model.tflite'
model = r'D:\keyword_spotting\server_models\custom_model_4\trail_1\16k_melspec-nfft-1024_a_cnn_dense_model.keras'

# lite_model = "/shareddrive/working/model_code/models/custom_model_4/trail_3/16k_melspec-nfft-1024_a_cnn_dense_model.tflite"
# model = '/shareddrive/working/model_code/models/custom_model_4/trail_3/16k_melspec-nfft-1024_a_cnn_dense_model.keras'

# lite_model = r"C:\Users\muhammaddaniyal2\Desktop\keyword_spotting\models\a_model.tflite"
# model = r'C:\Users\muhammaddaniyal2\Desktop\keyword_spotting\models\a_model.keras'

# lite_model = '/workspaces/keyword_spotting/server_models/custom_model_4/trail_1/16k_melspec-nfft-1024_a_cnn_dense_model.tflite'
# model = '/workspaces/keyword_spotting/server_models/custom_model_4/trail_1/16k_melspec-nfft-1024_a_cnn_dense_model.keras'

class A_Predictor(Predictor): #custom model 1 - #Adele
    

    def __init__(self):
        super().__init__()

        self.lite_model = tf.lite.Interpreter(model_path=lite_model)
        self.model = tf.keras.models.load_model(model)
        self.lite_model.allocate_tensors()
        self.input_details = self.lite_model.get_input_details()
        self.output_details = self.lite_model.get_output_details()
        self.max_seconds = 1
        self.label = {
            0:'unknown',1:'adele'
        }

class _A_Predictor: #dummy
    
    def __init__(self):
        self.target_sr = 16000
    
    def predict(self,data,sr):
        y = np.random.random(16000)
        return y,('unknown',89.3)
