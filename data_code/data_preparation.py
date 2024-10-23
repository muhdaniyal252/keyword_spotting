import librosa
import soundfile as sf
import random
import os
from pydub import AudioSegment
from pydub.silence import split_on_silence
import wave
import pandas as pd
import shutil
from audiomentations import \
        ApplyImpulseResponse,\
            BitCrush,\
                BandPassFilter,\
                    BandStopFilter,\
                        Mp3Compression,\
                            GainTransition,\
                                Normalize,\
                                    HighShelfFilter,\
                                        LowShelfFilter,\
                                            PitchShift,\
                                                PolarityInversion,\
                                                    RoomSimulator,\
                                                        SevenBandParametricEQ


def remove_silence(source_folder,destination_folder,audio_name):

    def remove(input_file, output_file, min_silence_len=200, silence_thresh=-45):
        audio = AudioSegment.from_file(input_file)
        chunks = split_on_silence(audio, min_silence_len=min_silence_len,silence_thresh=silence_thresh)
        output = AudioSegment.empty()
        for chunk in chunks:
            output += chunk
        output.export(output_file,format='wav')

    for idx,file_name in enumerate(os.listdir(source_folder)):
        f = f'{audio_name}_{idx}.wav'
        source_file_path = os.path.join(source_folder, file_name)
        destination_file_path = os.path.join(destination_folder, f)
        remove(source_file_path,destination_file_path)

def remove_outliar(source_folder,destination_folder):

    def get_sample_rate(audio_path):
        with wave.open(audio_path, 'rb') as wav_file:
            sample_rate = wav_file.getframerate()
            num_frames = wav_file.getnframes()
            duration = num_frames / sample_rate
        return sample_rate,duration,audio_path
    
    audios = os.listdir(source_folder)
    audio_paths = [os.path.join(source_folder, file_name) for file_name in audios]
    audio_info = [get_sample_rate(a) for a in audio_paths]
    df = pd.DataFrame(audio_info, columns =['sample_rate', 'duration','audio_path'])
    min_duration = df['duration'].quantile(0.01)
    max_duration = df['duration'].quantile(0.99)
    print(min_duration,max_duration)
    new_df = df.loc[(df['duration']>=min_duration) & (df['duration']<=max_duration)]
    print(new_df.describe())
    destination_folder = '/'.join(destination_folder.split('/')[:-1])
    for pth in new_df['audio_path']:
        destination_file = f"{destination_folder}/{pth.split('/')[-1]}"
        shutil.copy2(pth,destination_file)

def augment(folder):

    def get_waveform(audio_path):
        y,sr = librosa.load(audio_path,sr=None)
        return y,sr

    def save_audio(audio_path,waveform,sr):
        sf.write(audio_path, waveform, sr)

    def apply_transform(transform,audio_paths,destination_folder_path,prefix):
        os.makedirs(destination_folder_path, exist_ok=True)
        for audio_path in audio_paths:
            audio_path = audio_path.replace('\\','/')
            waveform,sr = get_waveform(audio_path)
            augmented_audio = transform(waveform,sample_rate=sr)
            filename = f"{prefix}_{audio_path.split('/')[-1]}"
            augmented_audio_path = f"{destination_folder_path}/{filename}"
            save_audio(augmented_audio_path,augmented_audio,sr)

    _folder = f"{folder}/percentile"
    filepaths = [f"{_folder}/{f}" for f in os.listdir(_folder)]
    audio_file_paths = random.sample(filepaths,int(len(filepaths)))
    
    get_destination_folder = lambda destination: f"{folder}/{destination}"

    def pitch_emphasis():
        destination_folder = get_destination_folder('emphesis')
        transform = lambda x,sample_rate=None: librosa.effects.deemphasis(x,coef=0.8)
        apply_transform(transform,filepaths,destination_folder,'pe')
        return len(filepaths)

    def pitch_negative_emphesis():
        destination_folder = get_destination_folder('neg_emphesis')
        transform = lambda x,sample_rate=None: librosa.effects.preemphasis(x,coef=0.8)
        apply_transform(transform,filepaths,destination_folder,'pne')
        return len(filepaths)

    def bit_crush():
        destination_folder = get_destination_folder('bit_crush')
        transform = BitCrush(min_bit_depth=5, max_bit_depth=14, p=1.0)
        apply_transform(transform,filepaths,destination_folder,'bc')
        return len(filepaths)

    def band_pass():
        destination_folder = get_destination_folder('band_pass')
        transform = BandPassFilter(min_center_freq=100.0, max_center_freq=6000, p=1.0)
        apply_transform(transform,filepaths,destination_folder,'bp')
        return len(filepaths)

    def band_stop():
        destination_folder = get_destination_folder('band_stop')
        transform = BandStopFilter(min_center_freq=100.0, max_center_freq=6000, p=1.0)
        apply_transform(transform,filepaths,destination_folder,'bs')
        return len(filepaths)

    def mp3_compression():
        destination_folder = get_destination_folder('mp3_compression')
        transform = Mp3Compression(p=1.0)
        apply_transform(transform,filepaths,destination_folder,'mp3c')
        return len(filepaths)

    def pitch_shift():
        destination_folder = get_destination_folder('pitch_shift')
        transform = PitchShift(
            min_semitones=-5.0,
            max_semitones=5.0,
            p=1.0
        )
        apply_transform(transform,filepaths,destination_folder,'ps')
        return len(filepaths)

    def polarity_inversion():
        destination_folder = get_destination_folder('polarity_inversion')
        transform = PolarityInversion(p=1.0)
        apply_transform(transform,filepaths,destination_folder,'pi')
        return len(filepaths)

    def room_simulator():
        destination_folder = get_destination_folder('room_simulator')
        transform = RoomSimulator(p=1.0)
        apply_transform(transform,filepaths,destination_folder,'rs')
        return len(filepaths)

    def low_shelf_filter():
        destination_folder = get_destination_folder('low_shelf_filter')
        transform = LowShelfFilter(p=1.0)
        apply_transform(transform,filepaths,destination_folder,'lsf')
        return len(filepaths)

    def high_shelf_filter():
        destination_folder = get_destination_folder('high_shelf_filter')
        transform = HighShelfFilter(p=1.0)
        apply_transform(transform,filepaths,destination_folder,'hsf')
        return len(filepaths)

    def gain_transition():
        destination_folder = get_destination_folder('gain_transition')
        transform = GainTransition(p=1.0)
        apply_transform(transform,filepaths,destination_folder,'gt')
        return len(filepaths)

    def normalize():
        destination_folder = get_destination_folder('normalize')
        transform = Normalize(p=1.0)
        apply_transform(transform,filepaths,destination_folder,'n')
        return len(filepaths)

    def seven_band_parameter():
        destination_folder = get_destination_folder('seven_band_parameter')
        transform = SevenBandParametricEQ(p=1.0)
        apply_transform(transform,filepaths,destination_folder,'sbp')
        return len(filepaths)

    transforms = {
        'pitch_emphasis':pitch_emphasis,
        'pitch_negative_emphesis':pitch_negative_emphesis,
        'bit_crush':bit_crush,
        'band_pass':band_pass,
        'band_stop':band_stop,
        'mp3_compression':mp3_compression,
        'gain_transition':gain_transition,
        'high_shelf_filter':high_shelf_filter,
        'low_shelf_filter':low_shelf_filter,
        'pitch_shift':pitch_shift,
        'polarity_inversion':polarity_inversion,
        'room_simulator':room_simulator,
        'normalize':normalize,
        'seven_band_parameter':seven_band_parameter
    }
    
    for transform,transform_fn in transforms.items():
        number_of_files = transform_fn()
        print(transform,number_of_files)

if __name__ == '__main__':
    
    main_folder = '/shareddrive/working/data_code/data/hilfe_hilfe'
    keyword = 'Hilfe-Hilfe'
    path_to_original = f'{main_folder}/original/'

    path_to_trimmed = f'{main_folder}/trimmed/'
    os.makedirs(path_to_trimmed,exist_ok=True)
    remove_silence(path_to_original,path_to_trimmed,keyword)

    path_to_percentile = f'{main_folder}/augmented/percentile/'
    os.makedirs(path_to_percentile,exist_ok=True)
    remove_outliar(path_to_trimmed,path_to_percentile)

    path_to_augment = f'{main_folder}/augmented/'
    os.makedirs(path_to_augment,exist_ok=True)
    augment(path_to_augment)
