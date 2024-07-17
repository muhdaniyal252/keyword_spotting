import tensorflow as tf
import numpy as np
import pickle
from audio_feature_extractors import *

class DataSequence(tf.keras.utils.Sequence):

    def __init__(self,data,batch_size):
        self.data_size = len(data)
        audios,labels = list(),list()
        for audio,label in data:
            audios.append(audio)
            labels.append(label)
        self.X = np.array(audios)
        self.Y = np.array(labels)
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(self.data_size / self.batch_size))

    def __getitem__(self,idx):
        s = idx * self.batch_size
        e = (idx + 1) * self.batch_size
        X = self.X[s:e]
        Y = self.Y[s:e]
        return X,Y

class DataSequenceRaw(tf.keras.utils.Sequence):

    def __init__(self,data,batch_size,feature_extractor='mfcc',max_seconds=2,sr=None):
        self.data_size = len(data)
        audios,labels = list(),list()
        for audio,label in data:
            audios.append(audio)
            labels.append(label)
        self.X = audios
        self.Y = np.array(labels)
        self.batch_size = batch_size
        self.feature_extractor = feature_extractor
        self.max_seconds = max_seconds
        self.sr = sr

        
    def __len__(self):
        return int(np.ceil(self.data_size / self.batch_size))

    def __getitem__(self,idx):
        s = idx * self.batch_size
        e = (idx + 1) * self.batch_size
        x = self.X[s:e]
        X = self.get_processed_data(x)
        Y = self.Y[s:e]
        return X,Y

    def get_processed_data(self,audio_files):
        data = list()
        for audio_file in audio_files:
            y,sr = librosa.load(audio_file,sr=self.sr)
            y = pad_or_trunc(y,sr*self.max_seconds)
            features = feature_extractors.get(self.feature_extractor)(y,sr)
            data.append(features)
        return np.array(data)
        
def _get_train_val_size(total_examples,train_percent=70):
    
    train_ratio = round(train_percent)/100
    remaining_percent = 100 - train_percent
    test_percent = round(remaining_percent/3)*2
    val_percent = remaining_percent - test_percent
    val_ratio = val_percent/100
    train_size = round(total_examples * train_ratio)
    val_size = round(total_examples * val_ratio)
    return train_size,val_size

def get_data(pickle_data_path,train_percent=70,batch_size=32):
    with open(pickle_data_path,'rb') as f:
        data = pickle.load(f)

    train_size,val_size = _get_train_val_size(len(data),train_percent)

    train_examples = data[:train_size]
    val_examples = data[train_size:train_size+val_size]
    test_examples = data[train_size+val_size:]

    train = DataSequence(train_examples,batch_size=batch_size)
    test = DataSequence(test_examples,batch_size=1)
    val = DataSequence(val_examples,batch_size=batch_size)

    return train,test,val

def get_data_raw(data,train_percent=70,batch_size=32,featuer_extractor='mfcc',sr=None,max_seconds=2):

    train_size,val_size = _get_train_val_size(len(data),train_percent)

    train_examples = data[:train_size]
    val_examples = data[train_size:train_size+val_size]
    test_examples = data[train_size+val_size:]

    train = DataSequenceRaw(train_examples,batch_size=batch_size,feature_extractor=featuer_extractor,sr=sr,max_seconds=max_seconds)
    test = DataSequenceRaw(test_examples,batch_size=1,feature_extractor=featuer_extractor,sr=sr,max_seconds=max_seconds)
    val = DataSequenceRaw(val_examples,batch_size=batch_size,feature_extractor=featuer_extractor,sr=sr,max_seconds=max_seconds)

    return train,test,val
