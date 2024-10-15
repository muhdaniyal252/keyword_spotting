from flask import Flask, render_template, jsonify, request
import numpy as np
import shutil
from handler import Handler
from io import BytesIO
import soundfile as sf
import os
import time
from pydub import AudioSegment
from threading import Thread

app = Flask(__name__)

handler = Handler(root_path=app.root_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/set_sr', methods=['POST'])
def set_sr():
    handler.sr = int(request.form['sr'])
    handler._al = handler.sr//2
    handler._hl = int(handler.sr*2.5)//2
    handler.a_data = np.zeros(handler.sr)
    handler.h_data = np.zeros(int(handler.sr*2.5))
    return jsonify('SR saved successfully')

@app.route('/upload', methods=['POST'])
def upload():
    audio = request.files['audio']
    audio_bytes = BytesIO(audio.read())
    audio_bytes.seek(0)  
    audio_format = audio.content_type.split('/')[1]  
    if audio_format != 'wav':
        sound = AudioSegment.from_file(audio_bytes, format=audio_format)
        audio_bytes = BytesIO()
        sound.export(audio_bytes, format='wav')
        audio_bytes.seek(0)
    new_data, _ = sf.read(audio_bytes)
    handler.a_audio_data = np.append(handler.a_audio_data,new_data)
    handler.h_audio_data = np.append(handler.h_audio_data,new_data)
    handler.progress = 0
    handler.total_items = 0
    total_items = handler.process()
    return jsonify({'total_items':total_items})

@app.route('/clear', methods=['POST'])
def clear():
    handler.reset()
    return jsonify({'status':'success'})

@app.route('/get_result',methods=['POST','GET'])
def get_result():
    if handler.completed:
        r = handler.results.copy()
        r.reverse()
        handler.results = []
    else:
        r = [handler.progress]
    return jsonify({'result':r})

@app.route('/move', methods=['POST'])
def move():
    path = f'{app.root_path}/{request.form["path"]}'
    label = request.form['label']
    handler.move(path, label)
    return jsonify({'status':'success'})

def remove_audios():
    while True:
        try:
            dest = os.path.join(app.root_path,'static','audios')
            shutil.rmtree(dest)
        except: pass
        finally:
            time.sleep(600)


if __name__ == '__main__':
    # Thread(target=remove_audios).start()
    # app.run(debug=False,host='0.0.0.0',port=5000)
    app.run(debug=False,host='0.0.0.0',port=5000,ssl_context='adhoc')