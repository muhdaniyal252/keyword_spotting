import requests
from collections import Counter
import glob

# Replace 'YOUR_DEEPGRAM_API_KEY' with your actual Deepgram API key
DEEPGRAM_API_KEY = '4207b8a639744fbdbe634616684bf7d67c8791e2'
AUDIO_FILE_PATH = r"D:\data_code\data\adele\trimmed\*.wav"  # Path to your audio file


# Set the parameters for the API request
url = 'https://api.deepgram.com/v1/listen'
headers = {
    'Authorization': f'Token {DEEPGRAM_API_KEY}',
    'Content-Type': 'audio/wav'
}
params = {
    'language': 'de',  # Specify the language as German
    'tier': 'enhanced',  # Use the enhanced tier for better accuracy
    'model': 'general'  # Specify the model
}
d = list()
for i in glob.glob(AUDIO_FILE_PATH):
    # Read the audio file
    with open(i, 'rb') as audio_file:
        response = requests.post(url, headers=headers, params=params, data=audio_file)

    print(i)
    # Check the response
    if response.status_code == 200:
        d.append(
            response.json()['results']['channels'][0]['alternatives'][0]['transcript']
        )
    else:
        print("Error:", response.json())

c = Counter(d)
print(c)