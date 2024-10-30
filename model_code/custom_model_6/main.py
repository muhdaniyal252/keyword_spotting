#!/usr/bin/env python
# coding: utf-8

# from mobile_net import get_model
import librosa
import tensorflow as tf
import glob
import numpy as np
import random
import wave
import os
import pickle
# import noisereduce as nr
import requests


gpus = tf.config.list_physical_devices('GPU')
gpu = gpus[0]
tf.config.experimental.set_memory_growth(gpu, True)
tf.config.list_logical_devices('GPU')




noise_path = '/shareddrive/working/data_code/data/neg_data/_background_noise_/chunks/1s_chunks/*'
environment_path = '/shareddrive/working/data_code/data/neg_data/envornment/chunks/1s_chunks/*'
word_path = '/shareddrive/working/data_code/data/neg_data/spcmd/all_words/original/*'
recording_path = '/shareddrive/working/data_code/data/neg_data/internet_recordings/chunks/1s_chunks/*'



word_files = glob.glob(word_path)
environment_files = glob.glob(environment_path)
noise_files = glob.glob(noise_path)
recording_files = glob.glob(recording_path)
len(environment_files),len(noise_files),len(word_files),len(recording_files)



def get_duration(audio_path):
    with wave.open(audio_path, 'rb') as wav_file:
        sample_rate = wav_file.getframerate()
        num_frames = wav_file.getnframes()
        duration = num_frames / sample_rate
    return duration



limit = 1.5
adele_path = '/shareddrive/working/data_code/data/adele/augmented/*'
adele_folders = glob.glob(adele_path)
adele_files = [j for i in adele_folders for j in glob.glob(f'{i}/*')]
# adele_files = [j for i in adele_folders for j in glob.glob(f'{i}/*') if get_duration(j) <= limit]
# hilfe_path = '/shareddrive/working/data_code/data/hilfe/augmented/*'
# hilfe_folders = glob.glob(hilfe_path)
# hilfe_files = [j for i in hilfe_folders for j in glob.glob(f'{i}/*')]
# # hilfe_files = [j for i in hilfe_folders for j in glob.glob(f'{i}/*') if get_duration(j) <= limit]


len(adele_files)#,len(hilfe_files)


# DEEPGRAM_API_KEY = '4207b8a639744fbdbe634616684bf7d67c8791e2'
# url = 'https://api.deepgram.com/v1/listen'
# headers = {
#     'Authorization': f'Token {DEEPGRAM_API_KEY}',
#     'Content-Type': 'audio/wav'
# }
# params = {
#     'language': 'de',  # Specify the language as German
#     'tier': 'enhanced',  # Use the enhanced tier for better accuracy
#     'model': 'general'  # Specify the model
# }
# ad_files = list()
# for i in adele_files:
#     # Read the audio file
#     with open(i, 'rb') as audio_file:
#         response = requests.post(url, headers=headers, params=params, data=audio_file)

#     if response.status_code == 200:
#         if 'adele' in response.json()['results']['channels'][0]['alternatives'][0]['transcript']:
#             ad_files.append(i)
#     else:
#         print("Error:", response.json())
# len(ad_files)



no_of_files = 1109+660
file_path_and_labels = list()
# file_path_and_labels.extend([(i,1) for i in random.sample(adele_files,no_of_files)])
file_path_and_labels.extend([(i,1) for i in adele_files])
# file_path_and_labels.extend([(i,2) for i in random.sample(hilfe_files,no_of_files)])
# file_path_and_labels.extend([(i,2) for i in hilfe_files])
# avg_files = (len(adele_files) + len(hilfe_files)) // 2
avg_files = len(adele_files)
# file_path_and_labels.extend([(i,0) for i in random.sample(recording_files,len(recording_files)//2)])
file_path_and_labels.extend([(i,0) for i in random.sample(word_files,avg_files)])
file_path_and_labels.extend([(i,0) for i in noise_files])
file_path_and_labels.extend([(i,0) for i in environment_files])
random.shuffle(file_path_and_labels)
avg_files



sr = 16000
max_seconds = 1
pad_or_trunc = lambda a,i : a[0:i] if len(a) > i else a if len(a) == i else np.pad(a,(0, (i-len(a))))

def process_data(y,sr,max_seconds):
    y = pad_or_trunc(y,sr*max_seconds)
    features = librosa.feature.melspectrogram(y=y,sr=sr,n_fft=1024)
    return features
    
def get_processed_data(audio_file):
    y,_ = librosa.load(audio_file,sr=sr)
    # y = nr.reduce_noise(y=y, sr=sr,n_fft=1024)
    # y[np.isnan(y)] = 0
    features = process_data(y,sr,max_seconds)
    return features
try:
    with open('f_and_ls.pickle','rb') as f:
        features_and_labels = pickle.load(f)
except:
    features_and_labels = list()
    for i,j in file_path_and_labels:
        try:
            features = get_processed_data(i)
            features_and_labels.append((features,j))
        except: print(i)
        # features = get_processed_data(i)
        # features_and_labels.append(features,j)
    # with open('f_and_l.pickle','wb') as f:
    #     pickle.dump(features_and_labels,f)
finally:
    print(features_and_labels[0][0].shape)




class DataSequenceRaw(tf.keras.utils.Sequence):

    def __init__(self,data,batch_size):
        self.data_size = len(data)
        audios,labels = zip(*data)
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


def _get_train_val_size(total_examples,train_percent=70):
    
    train_ratio = round(train_percent)/100
    remaining_percent = 100 - train_percent
    test_percent = round(remaining_percent/3)*2
    val_percent = remaining_percent - test_percent
    val_ratio = val_percent/100
    train_size = round(total_examples * train_ratio)
    val_size = round(total_examples * val_ratio)
    return train_size,val_size

def get_data_raw(data,train_percent=70,batch_size=32):

    train_size,val_size = _get_train_val_size(len(data),train_percent)

    train_examples = data[:train_size]
    val_examples = data[train_size:train_size+val_size]
    test_examples = data[train_size+val_size:]

    train = DataSequenceRaw(train_examples,batch_size=batch_size)
    test = DataSequenceRaw(test_examples,batch_size=1)
    val = DataSequenceRaw(val_examples,batch_size=batch_size)

    return train,test,val

train,test,val = get_data_raw(features_and_labels,train_percent=80)


shape = train[0][0][0].shape
input_shape = [*shape,1]


def get_model(
        input_shape,
        output_neurons=1,
        output_activation='sigmoid',
        loss=tf.keras.losses.binary_crossentropy,
        lr=0.0001
):
    _input = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(512,kernel_size=3,padding='valid',activation='relu')(_input)
    x = tf.keras.layers.Conv2D(256,kernel_size=3,padding='valid',activation='relu')(x)
    x = tf.keras.layers.MaxPool2D((2,2))(x)
    x = tf.keras.layers.Conv2D(128,kernel_size=3,padding='valid',activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Conv2D(128,kernel_size=3,padding='valid',activation='relu')(x)
    x = tf.keras.layers.MaxPool2D((2,2))(x)
    x = tf.keras.layers.Conv2D(64,kernel_size=3,padding='valid',activation='relu')(x)
    x = tf.keras.layers.MaxPool2D((2,2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1024,activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(1024,activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(1024,activation='relu')(x)
    x = tf.keras.layers.Dropout(0.7)(x)
    x = tf.keras.layers.Dense(1024,activation='relu')(x)
    x = tf.keras.layers.Dense(10,activation='relu')(x)
    outputs = tf.keras.layers.Dense(output_neurons,activation=output_activation,kernel_regularizer=tf.keras.regularizers.L2(l2=0.01))(x)
    model = tf.keras.Model(inputs=_input,outputs=outputs)

    model.compile(
        loss=loss,
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        metrics=['accuracy'],
    )

    return model

model = get_model(
        input_shape=input_shape,
        lr=0.001
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy',factor=0.1,patience=5,mode='max')
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',patience=1,mode='max',restore_best_weights=True,start_from_epoch=10)
with tf.device('/gpu'):
    history = model.fit(train,epochs=100,validation_data=val,verbose=1,callbacks=[reduce_lr,early_stopping])


model.evaluate(test)


folder_path = '/shareddrive/working/model_code/models/custom_model_4/trail_1'
os.makedirs(folder_path,exist_ok=True)
model_path = f'{folder_path}/16k_melspec-nfft-1024_a_cnn_dense_model.keras'
model.save(model_path)


