var recorder;

var audioContext = new AudioContext();
const constraints = { audio:true, video:false }
_xhr = new XMLHttpRequest();
_xhr.open('POST', '/set_sr', true);
_fd = new FormData()
_fd.append('sr',audioContext.sampleRate)
_xhr.send(_fd)
_xhr.onload = function(e){
    console.log('sr',e.target.response)
}


startButton.addEventListener("click", startRecording);
stopButton.addEventListener("click", stopRecording);
clearButton.addEventListener("click", clearResults);


function startRecording(){
    to_send = true;
    navigator.mediaDevices.getUserMedia(constraints).then(stream =>{
        input = audioContext.createMediaStreamSource(stream);
        recorder = new Recorder(input, {numChannels: 1});
        recorder.record();

        // setInterval(()=>{
        //     recorder.exportWAV(function(blob){
        //         audioChunks.push(blob);
        //     })
        //     recorder.clear();
        // }, recordMs);

    }).catch(error =>{
        console.error('Error accessing media devices.', error);
    });
}

function stopRecording(){
    recorder.stop();
    recorder.exportWAV(postRecordings)
    recorder.clear();
}

function postRecordings(blob){
    var xhr = new XMLHttpRequest();

    var fd = new FormData();
    fd.append('audio', blob, 'recorded_audio');
    xhr.open("POST", "/upload", true);
    xhr.send(fd)
    xhr.onload = function(e){
        if (this.readyState === 4){
            if (this.status == 200){
                result = JSON.parse(e.target);
                console.log(result);
            }
        }
    }
}

function move(path,prediction,word_model,result){
    var xhr = new XMLHttpRequest();
    xhr.open("POST", "/move", true);
    if(result == 'correct'){
        var label = prediction;
    } else if (result == 'incorrect') {
        if (prediction == 'unknown'){
            var label = word_model;
        } else {
            var label = 'unknown';
        }
    }
    var fd = new FormData();
    fd.append('path', path);
    fd.append('label', label);
    xhr.send(fd);
}

function getResults(){
    function getResult(){
        fetch('/get_result')
        .then(response => response.json())
        .then(function(obj){
            if (obj['result']){
                var newItem = document.createElement('tr');
                if (obj['result']['prediction'] == 'adele'){
                    newItem.classList.add('table-success');
                } else if (obj['result']['prediction'] == 'hilfe') {
                    newItem.classList.add('table-info');
                }

                var newCell = document.createElement('td');
                var audio = document.createElement('audio');
                var source = document.createElement('source');
                source.src = obj['result']['path'];
                source.type = 'audio/wav';
                audio.controls = true;
                audio.innerHTML = 'Your browser does support audio!'
                audio.appendChild(source);
                newCell.appendChild(audio);
                newItem.appendChild(newCell);

                var newCell = document.createElement('td');
                newCell.innerHTML = obj['result']['prediction'];
                newItem.appendChild(newCell);

                var newCell = document.createElement('td');
                newCell.innerHTML = obj['result']['l_prediction'];
                newItem.appendChild(newCell);
                
                var newCell = document.createElement('td');
                newCell.innerHTML = obj['result']['s_prediction'];
                newItem.appendChild(newCell);

                var correctButton = document.createElement('a');
                var inCorrectButton = document.createElement('a');

                var newCell = document.createElement('td');
                correctButton.classList.add('btn');
                correctButton.classList.add('btn-outline-success');
                correctButton.onclick = function(){
                    correctButton.classList.remove('btn-outline-success');
                    correctButton.classList.add('btn-success');
                    correctButton.style.pointerEvents = 'none';
                    inCorrectButton.style.pointerEvents = 'none';
                    move(obj['result']['path'],obj['result']['prediction'],obj['result']['word_model'],'correct');
                };
                var correctIcon = document.createElement('i');
                correctIcon.classList.add('fa-solid');
                correctIcon.classList.add('fa-check');
                correctButton.appendChild(correctIcon);
                newCell.appendChild(correctButton);
                newItem.appendChild(newCell);

                var newCell = document.createElement('td');
                inCorrectButton.classList.add('btn');
                inCorrectButton.classList.add('btn-outline-danger');
                inCorrectButton.onclick = function(){
                    inCorrectButton.classList.remove('btn-outline-danger');
                    inCorrectButton.classList.add('btn-danger');
                    correctButton.style.pointerEvents = 'none';
                    inCorrectButton.style.pointerEvents = 'none';
                    move(obj['result']['path'],obj['result']['prediction'],obj['result']['word_model'],'incorrect');
                };
                var correctIcon = document.createElement('i');
                correctIcon.classList.add('fa-solid');
                correctIcon.classList.add('fa-xmark');
                inCorrectButton.appendChild(correctIcon);
                newCell.appendChild(inCorrectButton);
                newItem.appendChild(newCell);
                
                labels.insertBefore(newItem, labels.firstChild);
                
                var newCell = document.createElement('td');
                var anchor = document.createElement('a');
                anchor.href = obj['result']['path'];
                n = obj['result']['path'].split('/')
                anchor.innerHTML = n[n.length-1];
                anchor.download = n[n.length-1];
                newCell.appendChild(anchor);
                newItem.appendChild(newCell);
            }
        });
    }
    setInterval(getResult, 100);
}


function clearResults(){
    labels.innerHTML = '';
    var xhr = new XMLHttpRequest();
    xhr.open("POST", "/clear", true);
    xhr.send();
}

postRecordings()
getResults()
