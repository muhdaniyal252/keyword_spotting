import os
import azure.cognitiveservices.speech as speechsdk
from synthesizer import Synthesizer
import io
from scipy.io.wavfile import write
os.environ['SPEECH_KEY'] = '8cc6ab4e7b4940509c2d7d1e6cf6c224'
os.environ['SPEECH_REGION'] = 'germanywestcentral'

# class MicroSoft:
class MicroSoft(Synthesizer):

    def __init__(self) -> None:
        self.speech_config = speechsdk.SpeechConfig(subscription=os.environ.get('SPEECH_KEY'), region=os.environ.get('SPEECH_REGION'))
        self.speech_config.speech_recognition_language = "de-DE"  # Set to German


    def synthesize(self, audio_array, sr):
        byte_io = io.BytesIO()
        write(byte_io, sr, audio_array)
        byte_io.seek(0)
        push_stream = speechsdk.audio.PushAudioInputStream()
        push_stream.write(byte_io.read())
        push_stream.close()
        audio_config = speechsdk.audio.AudioConfig(stream=push_stream)
        speech_recognizer = speechsdk.SpeechRecognizer(speech_config=self.speech_config, audio_config=audio_config)
        speech_recognition_result = speech_recognizer.recognize_once_async().get()
        return speech_recognition_result.text
