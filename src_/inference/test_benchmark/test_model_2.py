import os
import librosa
import pandas as pd
import torch
import json
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

def predict_labels_for_chunks(models, chunk_df):
    for model_num, model in enumerate(models, start=1):
        predictions = []

        for index, row in chunk_df.iterrows():
            file_path = row['Path']
            waveform, _ = librosa.load(file_path, sr=44100)
            predicted_label = keyword_spotting(model, waveform)
            predictions.append(predicted_label)

        chunk_df[f"Model_{model_num}_predicted_label"] = predictions

    return chunk_df

if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load models
    models = []
    for i in range(1, 4):
        model_dir = f"/home/majam001/kws/alpha-kws/models/pless_sweep_mfcc40_5/model_{i}/learned_net"
        model_config_path = os.path.join(model_dir, 'net.config')
        model_checkpoint_path = os.path.join(model_dir, 'checkpoint/checkpoint.pth.tar')

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

    # Load the existing Excel file
    excel_file_path = "/home/majam001/kws/alpha-kws/src/inference/test_benchmark/test_benchmark.xlsx"
    chunk_df = pd.read_excel(excel_file_path)

    # Predict labels for chunks and append to DataFrame
    chunk_df = predict_labels_for_chunks(models, chunk_df)

    # Save the updated DataFrame to the Excel file
    chunk_df.to_excel(excel_file_path, index=False)
