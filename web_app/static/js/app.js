let audioChunks = [];
let to_send = false;
const recordMs = 5000;
var recorder;

var audioContext = new AudioContext();
const constraints = { audio: true, video:false }
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

        setInterval(()=>{
            recorder.exportWAV(function(blob){
                audioChunks.push(blob);
            })
            recorder.clear();
        }, recordMs);

    }).catch(error =>{
        console.error('Error accessing media devices.', error);
    });
}

function stopRecording(){
    to_send = false;
    recorder.clear();
    recorder.stop();
    audioChunks = [];
}

function postRecordings(){
    var xhr = new XMLHttpRequest();

    var fd = new FormData();
    fd.append('audio', null);

    xhr.onreadystatechange = function() {
        if (xhr.readyState == 4 && xhr.status == 200) {
            console.log('sent')
        }
    }
    setInterval(() => {
        if (to_send && audioChunks.length){
            xhr.open("POST", "/upload", true);
            audio = audioChunks.shift();
            fd.set('audio',audio)
            xhr.send(fd)
        } 
    }, 1000);
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

// function move(path,label){
//     var xhr = new XMLHttpRequest();
//     xhr.open("POST", "/move", true);
//     var fd = new FormData();
//     fd.append('path', path);
//     fd.append('label', label);
//     xhr.send(fd);
// }

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
                newCell.innerHTML = obj['result']['score'];
                newItem.appendChild(newCell);

                var newCell = document.createElement('td');
                newCell.innerHTML = obj['result']['prediction'];
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

// function getResults(){
//     function getResult(){
//         fetch('/get_result')
//         .then(response => response.json())
//         .then(function(obj){
//             console.log(obj);
//             if (obj['result']){
//                 var newItem = document.createElement('tr');
//                 if (obj['result']['prediction'] == 'adele'){
//                     newItem.classList.add('table-success');
//                 } else if (obj['result']['prediction'] == 'hilfe') {
//                     newItem.classList.add('table-info');
//                 }
//                 var newCell = document.createElement('th');
//                 newCell.scope = 'row';
//                 newCell.innerHTML = obj['result']['prediction'];
//                 newItem.appendChild(newCell);

//                 var newCell = document.createElement('td');
//                 newCell.innerHTML = obj['result']['score'];
//                 newItem.appendChild(newCell);

//                 var newCell = document.createElement('td');
//                 var audio = document.createElement('audio');
//                 var source = document.createElement('source');
//                 source.src = obj['result']['path'];
//                 source.type = 'audio/wav';
//                 audio.controls = true;
//                 audio.innerHTML = 'Your browser does support audio!'
//                 audio.appendChild(source);
//                 newCell.appendChild(audio);
//                 newItem.appendChild(newCell);

//                 var newCell = document.createElement('td');
//                 var correctButton = document.createElement('a');
//                 correctButton.classList.add('btn');
//                 correctButton.classList.add('btn-outline-success');
//                 correctButton.onclick = function(){
//                     move(obj['result']['path'],obj['result']['prediction']);
//                 };
//                 var correctIcon = document.createElement('i');
//                 correctIcon.classList.add('fa-regular');
//                 correctIcon.classList.add('fa-thumbs-up');
//                 correctButton.appendChild(correctIcon);
//                 newCell.appendChild(correctButton);
//                 newItem.appendChild(newCell);

//                 var newCell = document.createElement('td');
//                 var adeleButton = document.createElement('button');
//                 adeleButton.classList.add('btn');
//                 adeleButton.classList.add('btn-sm');
//                 adeleButton.classList.add('btn-outline-secondary');
//                 adeleButton.innerHTML = 'Adele'
//                 adeleButton.onclick = function(){
//                     move(obj['result']['path'],'adele');
//                 };
//                 newCell.appendChild(adeleButton);
//                 newItem.appendChild(newCell);

//                 var newCell = document.createElement('td');
//                 var hilfeButton = document.createElement('button');
//                 hilfeButton.classList.add('btn');
//                 hilfeButton.classList.add('btn-sm');
//                 hilfeButton.classList.add('btn-outline-secondary');
//                 hilfeButton.innerHTML = 'Hilfe'
//                 hilfeButton.onclick = function(){
//                     move(obj['result']['path'],'hilfe');
//                 };
//                 newCell.appendChild(hilfeButton);
//                 newItem.appendChild(newCell);

//                 var newCell = document.createElement('td');
//                 var unknownButton = document.createElement('button');
//                 unknownButton.classList.add('btn');
//                 unknownButton.classList.add('btn-sm');
//                 unknownButton.classList.add('btn-outline-secondary');
//                 unknownButton.innerHTML = 'Unknown'
//                 unknownButton.onclick = function(){
//                     move(obj['result']['path'],'unknown');
//                 };
//                 newCell.appendChild(unknownButton);
//                 newItem.appendChild(newCell);

//                 var newCell = document.createElement('td');
//                 var anchor = document.createElement('a');
//                 anchor.href = obj['result']['path'];
//                 n = obj['result']['path'].split('/')
//                 anchor.innerHTML = n[n.length-1];
//                 anchor.download = n[n.length-1];
//                 newCell.appendChild(anchor);
//                 newItem.appendChild(newCell);

//                 labels.insertBefore(newItem, labels.firstChild);
//             }
//         });
//     }
//     setInterval(getResult, 100);
// }

function clearResults(){
    labels.innerHTML = '';
    var xhr = new XMLHttpRequest();
    xhr.open("POST", "/clear", true);
    xhr.send();
}

postRecordings()
getResults()
