import os
import numpy as np
import librosa
import soundfile as sf
import torch
import json
import argparse
import pandas as pd
from models import get_net_by_name

# Function to perform keyword spotting
def keyword_spotting(model, waveform):
    with torch.no_grad():
        waveform = torch.from_numpy(waveform.astype('float32'))
        waveform = waveform.unsqueeze(0).unsqueeze(0)
        waveform = waveform.to(device)
        output = model(waveform)

    output = torch.nn.functional.softmax(output, dim=1)
    predicted_class_index = torch.argmax(output, dim=1).item()

    class_labels = ['adele', 'hilfe hilfe', 'unknown', 'silence']
    predicted_class_label = class_labels[predicted_class_index]

    return predicted_class_label

def predict_labels_for_chunks(models, chunk_dir):
    predictions = {f"Model_{i+1}_predicted_label": [] for i in range(len(models))}
    chunk_info = []

    for file_name in sorted(os.listdir(chunk_dir)):
        if file_name.endswith(".wav"):
            file_path = os.path.join(chunk_dir, file_name)
            waveform, _ = librosa.load(file_path, sr=44100)

            for i, model in enumerate(models):
                predicted_label = keyword_spotting(model, waveform)
                predictions[f"Model_{i+1}_predicted_label"].append(predicted_label)

            chunk_info.append((file_name, file_path))

    df = pd.DataFrame(chunk_info, columns=['Chunk Index', 'Path'])
    
    for model_num, preds in predictions.items():
        df[model_num] = preds
    
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='path to model dir', type=str, default=None)
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load models
    models = []
    for i in range(1, 4):
        model_dir = os.path.join(args.path, f"model_{i}")
        model_config_path = os.path.join(model_dir, 'learned_net/net.config')
        model_checkpoint_path = os.path.join(model_dir, 'learned_net/checkpoint/checkpoint.pth.tar')

        # Load model architecture
        net_config = json.load(open(model_config_path, 'r'))
        model = get_net_by_name(net_config['name']).build_from_config(net_config)
        model.to(device)

        # Load model weights
        checkpoint = torch.load(model_checkpoint_path, map_location=device)
        if 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
        model.load_state_dict(checkpoint)

        model.eval()
        models.append(model)

    # Set the path to the testing dataset
    chunk_dir = "/home/majam001/kws/alpha-kws/src/inference/test_benchmark/chunks"

    # Predict labels for the audio chunks and store the predictions in a DataFrame
    df = predict_labels_for_chunks(models, chunk_dir)

    # Append the DataFrame to the existing Excel file
    excel_file_path = "/home/majam001/kws/alpha-kws/src/inference/test_benchmark/test_benchmark.xlsx"
    if os.path.exists(excel_file_path):
        # Read the existing Excel file
        existing_df = pd.read_excel(excel_file_path)

        # Add the new predictions to the existing DataFrame
        existing_df = pd.concat([existing_df, df], axis=1)

        # Write the updated DataFrame back to the Excel file
        existing_df.to_excel(excel_file_path, index=False)
    else:
        df.to_excel(excel_file_path, index=False)