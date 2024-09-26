# from synthesizer import Synthesizer

# class MicroSoft(Synthesizer):

#     def __init__(self) -> None:
#         super().__init__()


#     def synthesize(self, *args, **kwargs):
#         return super().synthesize(*args, **kwargs)
    
import os
import azure.cognitiveservices.speech as speechsdk

os.environ['SPEECH_KEY'] = '8cc6ab4e7b4940509c2d7d1e6cf6c224'
os.environ['SPEECH_REGION'] = 'germanywestcentral'
import os
import numpy as np
import azure.cognitiveservices.speech as speechsdk
import io
def recognize_from_numpy(audio_array, sample_rate):
    # This example requires environment variables named "SPEECH_KEY" and "SPEECH_REGION"
    speech_config = speechsdk.SpeechConfig(subscription=os.environ.get('SPEECH_KEY'), region=os.environ.get('SPEECH_REGION'))
    speech_config.speech_recognition_language = "de-DE"  # Set to German

    # Convert NumPy array to bytes (int16 format)
    audio_bytes = audio_array.tobytes()  # Convert float32 to int16

    # Create a PushAudioInputStream
    push_stream = speechsdk.audio.PushAudioInputStream()

    # Write audio data to the stream
    push_stream.write(audio_bytes)
    push_stream.close()  # Close the stream after writing

    # Use the PushAudioInputStream as the audio input
    audio_config = speechsdk.audio.AudioConfig(stream=push_stream)
    
    # Create a SpeechRecognizer with the audio configuration
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    print("Recognizing from NumPy array...")
    
    # Recognize speech from the stream
    speech_recognition_result = speech_recognizer.recognize_once_async().get()

    if speech_recognition_result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print("Recognized: {}".format(speech_recognition_result.text))
    elif speech_recognition_result.reason == speechsdk.ResultReason.NoMatch:
        print("No speech could be recognized: {}".format(speech_recognition_result.no_match_details))
    elif speech_recognition_result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = speech_recognition_result.cancellation_details
        print("Speech Recognition canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print("Error details: {}".format(cancellation_details.error_details))
            print("Did you set the speech resource key and region values?")

audio_file_path = r'D:\keyword_spotting\demo_web\static\audios\2024-09-23_05-07\Hilfe-Hilfe\818a43fef2_Hilfe-Hilfe.wav'  # Replace with your local audio file path
import librosa

sample_rate = 16000
audio_array, sr = librosa.load(audio_file_path)

audio_array = librosa.resample(y=audio_array,orig_sr=sr,target_sr=sample_rate)

recognize_from_numpy(audio_array, sample_rate)

def recognize_from_file(audio_file_path):
    # This example requires environment variables named "SPEECH_KEY" and "SPEECH_REGION"
    speech_config = speechsdk.SpeechConfig(subscription=os.environ.get('SPEECH_KEY'), region=os.environ.get('SPEECH_REGION'))
    speech_config.speech_recognition_language = "de-DE"  # Set to German

    # Use the audio file instead of the default microphone
    audio_config = speechsdk.audio.AudioConfig(filename=audio_file_path)
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    print(f"Recognizing from file: {audio_file_path}")
    speech_recognition_result = speech_recognizer.recognize_once_async().get()

    if speech_recognition_result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print("Recognized: {}".format(speech_recognition_result.text))
    elif speech_recognition_result.reason == speechsdk.ResultReason.NoMatch:
        print("No speech could be recognized: {}".format(speech_recognition_result.no_match_details))
    elif speech_recognition_result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = speech_recognition_result.cancellation_details
        print("Speech Recognition canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print("Error details: {}".format(cancellation_details.error_details))
            print("Did you set the speech resource key and region values?")

# Specify the path to your audio file here
audio_file_path = 'path/to/your/audio/file.wav'  # Replace with your local audio file path
recognize_from_file(audio_file_path)