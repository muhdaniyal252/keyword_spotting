import os
import numpy as np
import torchaudio
import soundfile as sf
import librosa
import torch
import json
import argparse
import importlib
from models import get_net_by_name

# Function to create directories based on class labels
def create_output_directories(labels, base_dir="output"):
    for label in labels:
        os.makedirs(os.path.join(base_dir, label), exist_ok=True)

# Function to chunk the audio file
def chunk_audio_file(audio_file, chunk_size=2, overlap=1, save_dir="chunks"):
    os.makedirs(save_dir, exist_ok=True)

    waveform, sample_rate = librosa.load(audio_file, sr=44100)

    chunk_size_samples = int(chunk_size * sample_rate)
    overlap_samples = int(overlap * sample_rate)

    start = 0
    chunk_index = 1

    while start < len(waveform):
        end = start + chunk_size_samples
        chunk = waveform[start:end]

        if len(chunk) < chunk_size_samples:
            break

        # Save the chunk to a temporary WAV file
        temp_wav_path = os.path.join(save_dir, f"chunk_{chunk_index}.wav")
        sf.write(temp_wav_path, chunk, sample_rate, format='WAV')

        start += chunk_size_samples - overlap_samples
        chunk_index += 1

# Function to predict labels for audio chunks
def predict_labels_for_chunks(model, chunk_dir, output_dir="output"):
    for file_name in sorted(os.listdir(chunk_dir)):
        if file_name.endswith(".wav"):
            file_path = os.path.join(chunk_dir, file_name)
            waveform, _ = librosa.load(file_path, sr=44100)

            # Perform keyword spotting
            predicted_label = keyword_spotting(model, waveform)

            # Move the chunk to the corresponding output directory
            output_subdir = os.path.join(output_dir, predicted_label)
            output_path = os.path.join(output_subdir, file_name)

            os.makedirs(output_subdir, exist_ok=True)
            os.rename(file_path, output_path)

# Function to perform keyword spotting
def keyword_spotting(model, waveform):
    # Perform inference with your model
    with torch.no_grad():
        waveform = torch.from_numpy(waveform.astype('float32'))
        waveform = waveform.unsqueeze(0).unsqueeze(0)
        waveform = waveform.to(device)
        output = model(waveform)

    output = torch.nn.functional.softmax(output, dim=1)
    predicted_class_index = torch.argmax(output, dim=1).item()

    class_labels = ['Adele', 'Hilfe Hilfe', 'unknown', 'silence']
    predicted_class_label = class_labels[predicted_class_index]

    return predicted_class_label

if __name__ == "__main__":
    torchaudio_enabled = False if importlib.util.find_spec('torchaudio') else False

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='path to model dir', type=str, default=None)
    args = parser.parse_args()

    # Load your trained PyTorch model checkpoint
    model_config_path = '%s/net.config' % args.path
    #trained_weights = args.ckp
    
    #best_model_path = '%s/checkpoint/model_best.pth.tar' % args.path
    best_model_path = '%s/checkpoint/checkpoint.pth.tar' % args.path
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if os.path.isfile(model_config_path):
        # load net from file
        net_config = json.load(open(model_config_path, 'r'))
        model = get_net_by_name(net_config['name']).build_from_config(net_config)
        #model = torch.nn.DataParallel(model)
        model.to(device)
        #print(model)

    checkpoint = torch.load(best_model_path, map_location=device)
    if 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
    
    model.load_state_dict(checkpoint)

    model.eval()
    
    # Set the path to your pre-recorded audio file
    audio_file_path = '/home/majam001/kws/alpha-kws/src/inference/test_audio.wav'

    # Set the chunk size and overlap
    chunk_size = 2
    overlap = 1

    # Set the output directory
    output_directory = "test_output"

    # Create output directories based on class labels
    class_labels = ['Adele', 'Hilfe Hilfe', 'unknown', 'silence']
    create_output_directories(class_labels, base_dir=output_directory)

    # Chunk the audio file
    chunk_dir = "chunks"
    chunk_audio_file(audio_file_path, chunk_size, overlap, chunk_dir)

    # Predict labels for the audio chunks and move them to corresponding output directories
    predict_labels_for_chunks(model, chunk_dir, output_dir=output_directory)
