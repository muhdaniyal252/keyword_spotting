import os
from pathlib import Path
import re
import hashlib
import wave
import struct
import shutil
import librosa
import soundfile

#dataset_url = 'http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz'
#dataset_name = 'speech_commands_v0.01'
keywords = ['Adele', 'Hilfe Hilfe']
unknowns = ['Hallo Alpha', 'bed','bird','cat','dog','down','eight','five','four','go','happy','house','left','marvin','nine','no','off','on','one','right','seven','sheila','six','stop','three','tree','two','up','wow','yes','zero']
background_folder_name = '_background_noise_'
output_path = '../../data/'


def which_set(filename, validation_percentage=10, testing_percentage=10):
    MAX_NUM_WAVS_PER_CLASS = 2 ** 27 - 1  # ~134M

    """Determines which data partition the file should belong to.

    We want to keep files in the same training, validation, or testing sets even
    if new ones are added over time. This makes it less likely that testing
    samples will accidentally be reused in training when long runs are restarted
    for example. To keep this stability, a hash of the filename is taken and used
    to determine which set it should belong to. This determination only depends on
    the name and the set proportions, so it won't change as other files are added.

    It's also useful to associate particular files as related (for example words
    spoken by the same person), so anything after '_nohash_' in a filename is
    ignored for set determination. This ensures that 'bobby_nohash_0.wav' and
    'bobby_nohash_1.wav' are always in the same set, for example.

    Args:
      filename: File path of the data sample.
      validation_percentage: How much of the data set to use for validation.
      testing_percentage: How much of the data set to use for testing.

    Returns:
      String, one of 'training', 'validation', or 'testing'.
    """
    base_name = os.path.basename(filename)
    # We want to ignore anything after '_nohash_' in the file name when
    # deciding which set to put a wav in, so the data set creator has a way of
    # grouping wavs that are close variations of each other.
    
    # how to check if a string contains a substring
    # https://stackoverflow.com/questions/3437059/does-python-have-a-string-contains-substring-method   
    #   
    if '_nohash_' in base_name:
        hash_name = re.sub(r'_nohash_.*$', '', base_name)
    else:
        hash_name = re.sub(r'_([^_]+)$', '', base_name)
    # This looks a bit magical, but we need to decide whether this file should
    # go into the training, testing, or validation sets, and we want to keep
    # existing files in the same set even if more files are subsequently
    # added.
    # To do that, we need a stable way of deciding based on just the file name
    # itself, so we do a hash of that and then use that to generate a
    # probability value that we use to assign it.
    hash_name_hashed = hashlib.sha1(hash_name.encode('utf-8')).hexdigest()
    print(hash_name_hashed)
    percentage_hash = ((int(hash_name_hashed, 16) %
                        (MAX_NUM_WAVS_PER_CLASS + 1)) *
                       (100.0 / MAX_NUM_WAVS_PER_CLASS))
    if percentage_hash < validation_percentage:
        result = 'validation'
    elif percentage_hash < (testing_percentage + validation_percentage):
        result = 'testing'
    else:
        result = 'training'
    return result

# resamples audio using librosa
def resample_file(input_file, sr=44100):
    y, sr = librosa.load(input_file, sr=sr)
    #librosa.resample(y, orig_sr=_, target_sr=sr)
    soundfile.write(input_file, y, sr)

def which_files(input_dir, partition):
    for root, dirs, files in os.walk(input_dir):
        for filename in files:
            print(filename)
            if filename.endswith('.wav') and which_set(filename) == partition:
                class_name = os.path.basename(root)
                if class_name in keywords or class_name == background_folder_name:
                    yield os.path.join(root, filename), os.path.join(partition, class_name, filename)
                elif class_name in unknowns:
                    fname = f'{os.path.splitext(filename)[0]}_{class_name}.wav'
                    yield os.path.join(root, filename), os.path.join(partition, 'unknown', fname)
                else:
                    assert RuntimeError(f'{filename} caused an error.')


if __name__ == '__main__':
    #os.chdir(output_path)

    input_dir = "/home/majam001/kws/good_commands_unified_plus"
    partitioned = output_path
    if True or not os.path.exists(output_path):
        for dataset_type in ['training', 'testing', 'validation']:
            print(f'Creating {dataset_type} dataset in {partitioned} ...')
            for file_src, file_dest in which_files(input_dir, dataset_type):
                print(file_src, file_dest)
                print(f'Copying {file_src} to {os.path.join(partitioned, file_dest)} ...')
                Path(os.path.dirname(os.path.join(partitioned, file_dest))).mkdir(parents=True, exist_ok=True)
                shutil.copyfile(file_src, os.path.join(partitioned, file_dest))
                resample_file(os.path.join(partitioned, file_dest))

            silence_save_path = f'{partitioned}/{dataset_type}/silence/'
            sr = 44100
            Path(silence_save_path).mkdir()
            print(f'Creating silence.wav in {silence_save_path} ...')
            with wave.open(silence_save_path + 'silence.wav', 'w') as wavfile:
                wavfile.setnchannels(1)
                wavfile.setframerate(sr)
                wavfile.setsampwidth(2)
                for i in range(sr):
                    data = struct.pack('<h', 0)
                    wavfile.writeframesraw(data)

    print("Done.")
