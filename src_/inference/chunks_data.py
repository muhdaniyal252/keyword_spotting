import os
import librosa
import soundfile as sf
import pandas as pd

def chunk_audio_file(audio_file, chunk_size=2, overlap=1, save_dir="test_benchmark/chunks"):
    os.makedirs(save_dir, exist_ok=True)

    waveform, sample_rate = librosa.load(audio_file, sr=None)

    chunk_size_samples = int(chunk_size * sample_rate)
    print(chunk_size_samples)
    overlap_samples = int(overlap * sample_rate)

    start = 0
    chunk_index = 362

    chunk_info = []

    while start + chunk_size_samples < len(waveform):
        end = start + chunk_size_samples
        chunk = waveform[start:end]

        # Save the chunk to a temporary WAV file
        temp_wav_path = os.path.join(save_dir, f"chunk_{chunk_index}.wav")
        sf.write(temp_wav_path, chunk, sample_rate, format='WAV')

        chunk_info.append((f"chunk_{chunk_index}", temp_wav_path, "hilfe hilfe"))

        start += chunk_size_samples - overlap_samples
        chunk_index += 1

    # Save chunk information to an Excel file
    df = pd.DataFrame(chunk_info, columns=['Chunk', 'Path', 'True label'])
    excel_file_path = os.path.join(save_dir, "test_benchmark.xlsx")

    # Check if the file already exists
    if os.path.exists(excel_file_path):
        # Read existing data from Excel file
        existing_df = pd.read_excel(excel_file_path)
        # Concatenate existing data with new data
        updated_df = pd.concat([existing_df, df], ignore_index=True)
        # Write the updated DataFrame back to the Excel file
        updated_df.to_excel(excel_file_path, index=False)
    else:
        df.to_excel(excel_file_path, index=False)

# Example usage:
audio_file = "/home/majam001/kws/alpha-kws/src/inference/test_benchmark/hilfe_hilfe.wav"
chunk_audio_file(audio_file, chunk_size=2, overlap=1, save_dir="/home/majam001/kws/alpha-kws/src/inference/test_benchmark/chunks")
