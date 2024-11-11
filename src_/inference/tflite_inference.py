import argparse
import gradio as gr
import librosa
import numpy as np
import tensorflow as tf
import soundfile as sf
import time

audio_buffer = []
results = []
prediction_interval = 1  # Adjust this value as needed (in seconds)
chunk_size = 1
overlap = 0.5
silence_threshold = 0.4

def keyword_spotting(waveform):
    """
    Performs keyword spotting on an audio waveform using a TensorFlow Lite model.

    Parameters:
    - waveform (numpy.ndarray): The audio waveform as a 1D numpy array.

    Returns:
    - predicted_class_label: The predicted class label indicating the recognized keyword.

    Notes:
    - The function assumes a constant sample rate of 44100 Hz and a fixed duration of 3 seconds for the input audio.
    - If the input waveform's length is less than the expected duration, it pads the waveform with zeros.
    - Extracts Mel-frequency cepstral coefficients (MFCC) features from the preprocessed waveform using the `extract_features` function.
    - Reshapes and expands the features to match the input shape expected by the TensorFlow Lite model.
    - Performs inference using a TensorFlow Lite model, applying softmax to obtain class probabilities.
    - Adjusts the predicted class index based on probability thresholds.
    - Returns the predicted class label from a predefined list of class labels.
    """

    class_labels = ['Adele', 'Hilfe Hilfe', 'unknown', 'silence']
    sample_rate = 44100 
    duration = 3

    #inference_rate = sample_rate * duration
    #n_samples = waveform.shape[0]

    # if n_samples == inference_rate:
    #     waveform = waveform
    # elif n_samples < inference_rate:
    #     pad_size = tf.maximum(0, inference_rate - tf.shape(waveform)[0])
    #     padded_audio = tf.pad(waveform, paddings=[[0, pad_size]])
    #     waveform = padded_audio
    #     waveform = waveform.numpy()
    #     print(waveform.shape)

    #waveform = extract_features(sample=waveform)

    #input shape (1,10,414)
    #waveform = np.reshape(waveform, (1,10,414))
    #print(interpreter.get_input_details()[0])
    #waveform = tf.expand_dims(waveform, 3)

    # Perform inference with TensorFlow Lite model
    waveform = tf.expand_dims(waveform, axis=0)
    print(waveform.shape)
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], waveform)
    interpreter.invoke()
    output = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
    print(output)
    output =  tf.nn.softmax(output)
    print("percent output: ", output)

    max_value = tf.reduce_max(output, axis=1)
    if max_value.numpy() < 0.5:
        predicted_class_index = 2
    else:
        # Get the predicted class index
        predicted_class_index = tf.argmax(output, axis=1).numpy().item()

        if predicted_class_index == 1 and max_value.numpy() < 0.9:
            predicted_class_index = 2

    # Get the predicted class label
    predicted_class_label = class_labels[predicted_class_index]
    print(predicted_class_label)
   
    return predicted_class_label

def chunk_audio(audio_file):
    """
    Processes audio data in chunks and performs keyword spotting on the accumulated audio buffer.

    Parameters:
    - audio_file (str or tuple): If str, it is the file path to the audio file. If tuple, it contains
      a tuple (sample_rate, waveform) where sample_rate is an integer and waveform is a numpy array.

    Returns:
    - result: The predicted class label indicating the recognized keyword or a message indicating
      that the function is waiting for more audio.

    Notes:as
    - If 'args.mic' is False, the function assumes 'audio_file' is a tuple containing sample rate and waveform.
    - If 'args.mic' is True, the function loads audio data from the specified file path.
    - Resamples the audio to 44100 Hz to ensure a consistent sample rate.
    - Appends the resampled audio to an audio buffer for continuous processing.
    - Periodically performs keyword spotting on the accumulated audio buffer.
    - If enough audio is accumulated, clears the buffer for the next prediction interval.
    """

    # global audio_buffer
    # # Load the audio data from the file
    # waveform, sample_rate = librosa.load(audio_file, sr=44100)

    # chunk_size = prediction_interval * sample_rate
    # if not args.mic:
    #     waveform = audio_file[1].astype(np.float32)
    #     sample_rate = audio_file[0]

    # resampled_audio = librosa.resample(y=np.array(waveform), orig_sr=sample_rate, target_sr=44100, fix=True)  # Resample to 44100 Hz

    # # Append the audio data to the buffer
    # audio_buffer.extend(resampled_audio)

    # if not args.mic:
    #     result = keyword_spotting(resampled_audio)
    #     return result
    
    # #Check if the buffer has accumulated enough audio for a prediction
    # if len(audio_buffer) == chunk_size:
    #     resampled_audio = librosa.resample(y=np.array(audio_buffer), orig_sr=sample_rate, target_sr=44100)  
    #     result = keyword_spotting(resampled_audio)

    #     # Clear the buffer for the next prediction interval
    #     audio_buffer = []
    # else:
    #     result = "Waiting for more audio..."

    # return result

    global audio_buffer
    # Initialize results list
    global results
    result = "Waiting"
    # Load the audio data from the file
    waveform, sample_rate = librosa.load(audio_file, sr=44100)

    # Append the audio data to the buffer
    audio_buffer.extend(waveform)

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

        # Check if the chunk is not silent (adjust the threshold as needed)
        if np.max(np.abs(chunk)) > silence_threshold:
            # Resample the chunk to the target sample rate
            resampled_chunk = librosa.resample(y=chunk, orig_sr=sample_rate, target_sr=44100)

            result = keyword_spotting(resampled_chunk)

            # Append the result and timestamp to the results list
            results.append([timestamp, result])
    results.sort(reverse=True)
    return result, results

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='path to model dir', type=str, default="/home/majam001/kws/alpha-kws/models/model_f1score/tf3/model_f1score_2_float16.tflite")
    parser.add_argument('--mic', help='True to choose inference via mic, False to upload file', default=True)
    args = parser.parse_args()

    # Load the TensorFlow Lite model
    interpreter = tf.lite.Interpreter(model_path=args.path)
    interpreter.allocate_tensors()
    #print(interpreter.get_input_details()[0])
    
    # Select if the inference needs to be done via microphone or via file upload
    if args.mic==True: 
        input_audio = gr.Audio(label="Audio Input", source="microphone", type="filepath", streaming=True, every=prediction_interval)
    else:
        input_audio = gr.Audio(label="Audio Input", source="upload", type="filepath")
    
    # Create a gradio interface which takes in input via mic or uploaded file, and passes it onto the chunk_audio function as an input and returns the predicted class
    iface = gr.Interface(
    fn=chunk_audio,
    inputs=input_audio,
    live=True,
    outputs=[
        gr.Label(),
        gr.Dataframe(headers=["Timestamp", "Prediction"],)
    ],
    title="Keyword Spotting",
    description="Detect keywords in live audio input.",
    )

    iface.launch(share=True)
