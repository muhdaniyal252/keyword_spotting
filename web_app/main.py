from flask import Flask, render_template, jsonify, request
import numpy as np
import shutil
from handler import Handler
from io import BytesIO
import soundfile as sf
import os
import time
from threading import Thread

app = Flask(__name__)

handler = Handler(root_path=app.root_path)
handler.start_process()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/set_sr', methods=['POST'])
def set_sr():
    handler.sr = int(request.form['sr'])
    handler._l = handler.sr//2
    handler.data = np.zeros(handler.sr)
    return jsonify('SR saved successfully')

@app.route('/upload', methods=['POST'])
def upload():
    audio = request.files['audio']
    audio_bytes = BytesIO(audio.read())
    audio_bytes.seek(0)  
    new_data, _ = sf.read(audio_bytes)
    handler.audio_data = np.append(handler.audio_data,new_data)
    return jsonify({'status':'success'})

@app.route('/clear', methods=['POST'])
def clear():
    handler.reset()
    return jsonify({'status':'success'})

@app.route('/get_result')
def get_result():
    r = handler.results.pop(0) if handler.results else None
    return jsonify({'result':r})

def remove_audios():
    while True:
        try:
            dest = os.path.join(app.root_path,'static','audios')
            shutil.rmtree(dest)
        except: pass
        finally:
            time.sleep(600)


if __name__ == '__main__':
    Thread(target=remove_audios).start()
    app.run(debug=False,host='0.0.0.0',port=5000)