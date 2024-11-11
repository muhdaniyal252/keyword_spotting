import requests
from bs4 import BeautifulSoup
from pydub import AudioSegment
import numpy as np
import json
import io
import urllib.parse
import ffmpeg

def extract_audio_stream_url(url):
    response = requests.get(url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        audio_element = soup.find('div', {'class': 'player'})['data-jsb']
        audio_data = json.loads(audio_element)
        audio_url = audio_data.get('media')

        # Fix the URL if it's a relative path
        if audio_url and not audio_url.startswith('http'):
            audio_url = urllib.parse.urljoin(url, audio_url)

        return audio_url
    else:
        print(f"Failed to retrieve the webpage. Status Code: {response.status_code}")
        return None

def download_audio(url):
    response = requests.get(url)

    if response.status_code == 200:
        audio_data = response.content
        return audio_data
    else:
        print(f"Failed to download audio. Status Code: {response.status_code}")
        return None

def save_as_wav(audio_data, output_file):
    try:
        # Decode MP3 data
        audio_segment = AudioSegment.from_mp3(io.BytesIO(audio_data))

        # Convert to WAV using ffmpeg
        audio, _ = (
            ffmpeg.input('pipe:0')
            .output(output_file, format='wav')
            .run(input=audio_segment.raw_data, capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg._run.Error as e:
        print(e.stderr.decode())
        raise e

def create_dataset(audio_file, output_folder, duration=3000):
    sound = AudioSegment.from_wav(audio_file)
    total_duration = len(sound)

    for i in range(0, total_duration, duration):
        segment = sound[i:i+duration]
        segment.export(f"{output_folder}/segment_{i//duration}.wav", format="wav")

if __name__ == "__main__":
    website_url = "https://www.inforadio.de/livestream/"
    audio_stream_url = extract_audio_stream_url(website_url)

    if audio_stream_url:
        audio_data = download_audio(audio_stream_url)

        if audio_data:
            output_file = "output.wav"
            save_as_wav(audio_data, output_file)
            
            output_folder = "./dataset"
            create_dataset(output_file, output_folder)
            print(f"Dataset created in '{output_folder}'")
        else:
            print("Failed to download audio.")
    else:
        print("Failed to extract audio stream URL.")
