import torch
import torchaudio
import gradio as gr
import librosa
import importlib
import os 
import json
import sys
import numpy as np
import argparse
import torch.nn.functional
import time
import soundfile as sf
from models import get_net_by_name


torchaudio_enabled = True if importlib.util.find_spec('torchaudio') else False
# Initialize variables for audio accumulation and prediction interval
audio_buffer = []
results = []
prediction_interval = 3  # Adjust this value as needed (in seconds)
chunk_size = 2  # Moving window size (in seconds)
overlap = 1  # Overlap between consecutive chunks (in seconds)
silence_threshold = 0.4

def extract_features(sample):
    torchaudio_enabled = False
    if torchaudio_enabled:
        melkwargs = {
            'win_length': 640,
            'hop_length': 320,
            'n_fft': 640,
        }
        mfcc = torchaudio.transforms.MFCC(44100, 10, melkwargs=melkwargs)(sample)
        return mfcc
    else:
        #print(sample)s
        mfcc = librosa.feature.mfcc(y=sample, sr=44100, n_mfcc=10, hop_length=320, n_fft=640)
        mfcc = torch.from_numpy(mfcc.astype('float32'))
        return torch.reshape(mfcc, (1, mfcc.shape[0], mfcc.shape[1]))

# def chunk_audio(audio_file):
#     global audio_buffer
#     #print(audio_file)
#     # Load the audio data from the file
#     waveform, sample_rate = librosa.load(audio_file, sr=None)
#     #waveform = audio_file[1].astype(np.float32)
#     #sample_rate = audio_file[0]
#     # Append the audio data to the buffer
#     audio_buffer.extend(waveform)
#     # resampled_audio = librosa.resample(y=np.array(waveform), orig_sr=sample_rate, target_sr=44100)  # Resample to 44100 Hz
#     # print(resampled_audio.shape)
#     # result = keyword_spotting(resampled_audio)
    
#     print(len(audio_buffer))
#     # Check if the buffer has accumulated enough audio for a prediction
#     if len(audio_buffer) >= prediction_interval * sample_rate:
#         # Concatenate and resample the audio in the buffer
#         #combined_audio = np.concatenate(audio_buffer)
#         resampled_audio = librosa.resample(y=np.array(audio_buffer), orig_sr=sample_rate, target_sr=44100)  # Resample to 44100 Hz

#         # Implement your keyword spotting logic here, using resampled_audio
#         # Replace this with your actual keyword detection code
#         result = keyword_spotting(resampled_audio)

#         # Clear the buffer for the next prediction interval
#         audio_buffer = []
#     else:
#         result = "Waiting for more audio..."

#     return result
def chunk_audio(audio_file):
    global audio_buffer
    # Initialize results list
    global results
    result = "Waiting"
    # Load the audio data from the file
    waveform, sample_rate = librosa.load(audio_file, sr=44100)

    # Append the audio data to the buffer
    audio_buffer.extend(waveform)

    # Load the existing audio file
    try:
        existing_audio, existing_sample_rate = sf.read("/home/majam001/kws/alpha-kws/src/inference/gradio_test_recordings/saved_audio.wav")
        #print(existing_sample_rate)
    except sf.LibsndfileError:
        existing_audio, existing_sample_rate = np.array([]), 44100

    # Concatenate the new audio data with the existing audio
    concatenated_audio = np.concatenate([existing_audio, waveform])

    # Save the concatenated audio to the original WAV file
    save_path = "/home/majam001/kws/alpha-kws/src/inference/gradio_test_recordings/saved_audio.wav"  # Provide your desired save path
    sf.write(save_path, concatenated_audio, 44100, format='WAV')

    # Calculate the chunk size and overlap in samples
    chunk_size_samples = int(chunk_size * sample_rate)
    overlap_samples = int(overlap * sample_rate)

    # Check if the buffer has accumulated enough audio for a prediction
    while len(audio_buffer) >= chunk_size_samples:
        # Extract a chunk of audio data for processing
        chunk = np.array(audio_buffer[:chunk_size_samples])

        # Shift the buffer by the amount of processed data
        audio_buffer = audio_buffer[chunk_size_samples - overlap_samples:]

        # Get the current timestamp
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())

        # # Save the audio chunk to a temporary WAV file
        # temp_wav_path = "/home/majam001/kws/alpha-kws/src/inference/gradio_test_recordings/temp_audio/temp_audio_" + timestamp + ".wav"
        # sf.write(temp_wav_path, chunk, 44100, format='WAV')

        # # Load the existing audio file
        # try:
        #     existing_audio, existing_sample_rate = sf.read("/home/majam001/kws/alpha-kws/src/inference/gradio_test_recordings/saved_audio.wav")
        #     print(existing_sample_rate)
        # except sf.LibsndfileError:
        #     existing_audio, existing_sample_rate = np.array([]), 44100

        # # Concatenate the new audio data with the existing audio
        # concatenated_audio = np.concatenate([existing_audio, chunk])

        # # Save the concatenated audio to the original WAV file
        # save_path = "/home/majam001/kws/alpha-kws/src/inference/gradio_test_recordings/saved_audio.wav"  # Provide your desired save path
        # sf.write(save_path, concatenated_audio, 44100, format='WAV')

        # Check if the chunk is not silent (adjust the threshold as needed)
        if np.max(np.abs(chunk)) > silence_threshold:
            # Resample the chunk to the target sample rate
            resampled_chunk = librosa.resample(y=chunk, orig_sr=sample_rate, target_sr=44100)

            # # Save the audio chunk to a temporary WAV file
            # temp_wav_path = "/home/majam001/kws/alpha-kws/src/inference/gradio_test_recordings/temp_audio_no_silence/temp_audio_no_silence_" + timestamp + ".wav"
            # sf.write(temp_wav_path, resampled_chunk, 44100, format='WAV')

            # # Load the existing audio file
            # try:
            #     existing_audio, existing_sample_rate = sf.read("/home/majam001/kws/alpha-kws/src/inference/gradio_test_recordings/saved_audio_no_silence.wav")
            #     print(existing_sample_rate)
            # except sf.LibsndfileError:
            #     existing_audio, existing_sample_rate = np.array([]), 44100

            # # Concatenate the new audio data with the existing audio
            # concatenated_audio = np.concatenate([existing_audio, resampled_chunk])

            # # Save the concatenated audio to the original WAV file
            # save_path = "/home/majam001/kws/alpha-kws/src/inference/gradio_test_recordings/saved_audio_no_silence.wav"  # Provide your desired save path
            # sf.write(save_path, concatenated_audio, 44100, format='WAV')

            result = keyword_spotting(resampled_chunk)

            # Append the result and timestamp to the results list
            results.append([timestamp, result])
    results.sort(reverse=True)
    return result, results

def keyword_spotting(waveform):
    class_labels = ['Adele', 'Hilfe Hilfe', 'unknown', 'silence']

    # if args.mic==False:
    #     waveform = torch.tensor(audio_input[1],dtype=torch.float32)
    #     sample_rate = 44100
    #     n_samples = audio_input[0]
    # else:
    #     waveform, sample_rate = librosa.load(audio_input, sr=44100)
    #     # np.save("test_inp.npy", waveform)
    #     # print(sample_rate)
    #     # waveform, sample_rate = librosa.load(audio_input, sr=44100)
    #     print(sample_rate)
    #     # # print(len(audio_input))
    #     # # print(audio_input[0])
    #     # waveform_up = waveform
    #     # np.save("test_out.npy", waveform_up)

    #     #loaded_audio = np.load('test_inp.npy')
    #     #print(loaded_audio.shape)
    #     #print(loaded_audio)
    #     #waveform = librosa.resample(y=audio_input[1].astype(np.float32), orig_sr=48000, target_sr=44100)
    #     #waveform_mic = waveform
    #     #np.save("test_mic.npy", waveform_mic)
    #     #waveform_up = np.load('test_out.npy')
    #     #print(waveform_up == waveform_mic)
    #     # waveform = audio_input[1].astype(np.float32)

    #     print(waveform.shape)
    #     n_samples = waveform.shape[0]
    #     waveform = torch.from_numpy(np.reshape(waveform, (1, n_samples)))

    # print(waveform.shape)
    # print(waveform)
    
    # if n_samples == sample_rate:
    #     waveform = waveform
    # elif n_samples < sample_rate:
    #     padded_waveform = torch.zeros([1, sample_rate])
    #     padded_waveform[0, 0:n_samples] = waveform[0]
    #     waveform = padded_waveform
    # #elif n_samples > sample_rate:
    #     #trimmed_waveform = np.random.randint(n_samples - sample_rate)
    #     #waveform = waveform[trimmed_waveform:trimmed_waveform + sample_rate]
    #     # Trim the audio using librosa.effects.trim
    #     # print(librosa.get_duration(y=waveform))
    #     # trimmed_waveform, index = librosa.effects.trim(y=waveform, top_db=2)
    #     # waveform = trimmed_waveform
    #     # print(librosa.get_duration(y=waveform))

    #waveform = extract_features(sample=waveform)
    #np.savetxt("/home/majam001/kws/alpha-kws/src/inference/test_sample.txt", waveform, fmt='%.2f')
    
    
    # Perform inference with your model
    with torch.no_grad():
        waveform = torch.from_numpy(waveform.astype('float32'))
        print(waveform.shape)
        waveform = waveform.unsqueeze(0).unsqueeze(0)
        output = model(waveform.to(device))
    print(output)
    output =  torch.nn.functional.softmax(output, dim=1)
    print("percent output: ", output)

    max_value = torch.max(output, dim=1)
    if max_value.values.item() < 0.5:
        predicted_class_index = 2
    else:
        # Get the predicted class index
        predicted_class_index = torch.argmax(output, dim=1).item()

    if predicted_class_index == 1:
        if max_value.values.item() < 0.9:
            predicted_class_index = 2
    print(predicted_class_index)

    # Get the predicted class label
    predicted_class_label = class_labels[predicted_class_index]
    
    print(predicted_class_label)
   
    return predicted_class_label

if __name__ == "__main__":
    torchaudio_enabled = False if importlib.util.find_spec('torchaudio') else False

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='path to model dir', type=str, default=None)
    parser.add_argument('--mic', help='True to choose inference via mic, False to upload file', default=True)
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

    import torch.onnx
    
    # # # # Dummy input data for the model
    # dummy_input = torch.randn(1, 1, 10, 414).cuda()  # Replace with your input shape
    # # dummy_input = torch.randn(1, 1, 44100)  # Batch size of 1, 1 channel, and audio length of 16000
    # torchscript_model = torch.jit.trace(model, dummy_input)

    # from torch.utils.mobile_optimizer import optimize_for_mobile
    # torchscript_model_optimized = optimize_for_mobile(torchscript_model)
    # torch.jit.save(torchscript_model_optimized, "/home/majam001/kws/alpha-kws/models/model_f1score/model_f1score_2_torchscript.pt")
    # torchscript_model_optimized._save_for_lite_interpreter("/home/majam001/kws/alpha-kws/models/model_f1score/model_f1score_2_torchscript_lite.ptl")
    # # # # Export the model to ONNX format
    # onnx_path = "/home/majam001/kws/alpha-kws/models/model_f1score/model_f1score_2.onnx"
    # #torch.onnx.export(model, dummy_input, onnx_path, keep_initializers_as_inputs=True, do_constant_folding=False, input_names = ['input'], output_names=['output'], verbose=True, dynamic_axes={'input': {2: 'audio_length'}}, opset_version=11)
    # torch.onnx.export(model, dummy_input, onnx_path, verbose=True, opset_version=11)

    if args.mic==True: 
        source = "microphone"
        type="numpy"
    else:
        source = "upload"
        type="filepath"
   
    iface = gr.Interface(
    fn=chunk_audio,
    inputs=gr.Audio(label="Audio Input", source="microphone", type="filepath", streaming=True, every=10),
    #inputs=gr.Audio(label="Audio Input", source="upload", type="filepath"),
    live=True,
    #outputs="text",
    #outputs=gr.Textbox(max_lines=100),
    outputs=[
        gr.Label(),
        gr.Dataframe(headers=["Timestamp", "Prediction"],)
    ],
    title="Keyword Spotting",
    description="Detect keywords in live audio input.",
    )

    iface.launch(share=True)