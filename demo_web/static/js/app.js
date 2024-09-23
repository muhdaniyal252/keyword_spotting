let totalItems = 0;
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
    navigator.mediaDevices.getUserMedia(constraints).then(stream =>{
        input = audioContext.createMediaStreamSource(stream);
        recorder = new Recorder(input, {numChannels: 1});
        recorder.record();

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
                var data = JSON.parse(e.target.response);
                totalItems = data.total_items;
                document.getElementById('total-items').innerText = `Total Items to Process: ${totalItems}`;
                document.getElementById('progress-section').style.display = 'block';
                updateProgress();
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


function updateProgress() {
    fetch('/get_result')
        .then(response => response.json())
        .then(data => {
            console.log(data)
            const progressPercentage = (data.result.progress / totalItems) * 100;
            document.getElementById('progress-bar').style.width = progressPercentage + '%';

            if (data.progress < totalItems) {
                setTimeout(updateProgress, 500);  // Poll every 500ms
            } 
            // else {
            //     // Processing done, populate the results
            //     document.getElementById('progress-section').style.display = 'none';
            //     data.result.map(populateResult);
            // }
        });
}

function populateResult(obj){
    
    var newItem = document.createElement('tr');
    if (obj['prediction'] == 'adele'){
        newItem.classList.add('table-success');
    } else if (obj['prediction'] == 'hilfe') {
        newItem.classList.add('table-info');
    }

    var newCell = document.createElement('td');
    var audio = document.createElement('audio');
    var source = document.createElement('source');
    source.src = obj['path'];
    source.type = 'audio/wav';
    audio.controls = true;
    audio.innerHTML = 'Your browser does support audio!'
    audio.appendChild(source);
    newCell.appendChild(audio);
    newItem.appendChild(newCell);

    var newCell = document.createElement('td');
    newCell.innerHTML = obj['prediction'];
    newItem.appendChild(newCell);

    var newCell = document.createElement('td');
    newCell.innerHTML = obj['l_prediction'];
    newItem.appendChild(newCell);
    
    var newCell = document.createElement('td');
    newCell.innerHTML = obj['s_prediction'];
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
        move(obj['path'],obj['prediction'],obj['word_model'],'correct');
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
        move(obj['path'],obj['prediction'],obj['word_model'],'incorrect');
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
    anchor.href = obj['path'];
    n = obj['path'].split('/')
    anchor.innerHTML = n[n.length-1];
    anchor.download = n[n.length-1];
    newCell.appendChild(anchor);
    newItem.appendChild(newCell);
}


function clearResults(){
    labels.innerHTML = '';
    var xhr = new XMLHttpRequest();
    xhr.open("POST", "/clear", true);
    xhr.send();
}

postRecordings()
getResults()
