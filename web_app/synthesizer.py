import requests
import io
from scipy.io.wavfile import write

class Synthesizer:

    def __init__(self):
        self.url = 'https://api.deepgram.com/v1/listen'
        DEEPGRAM_API_KEY = '4207b8a639744fbdbe634616684bf7d67c8791e2'
        self.headers = {
            'Authorization': f'Token {DEEPGRAM_API_KEY}',
            'Content-Type': 'audio/wav'
        }
        self.params = {
            'language': 'de',  # Specify the language as German
            'tier': 'enhanced',  # Use the enhanced tier for better accuracy
            'model': 'general'  # Specify the model
        }


    def synthesize(self,audio_array,sr):
        byte_io = io.BytesIO()
        write(byte_io, sr, audio_array)
        byte_io.seek(0)
        response = requests.post(self.url, headers=self.headers, params=self.params, data=byte_io)
        if response.status_code == 200:
            try:
                return response.json()['results']['channels'][0]['alternatives'][0]['transcript']
            except Exception as e:
                print(e)
                return None
        return None