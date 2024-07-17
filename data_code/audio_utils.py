import os
import wave

def get_all_audio_paths(keyword,number=1,real='3.1 clean'):
    augmentations = [
        'band_pass',
        'band_stop',
        'bit_crush',
        'emphesis',
        'gain_transition',
        'high_shelf_filter',
        'impulse',
        'low_shelf_filter',
        'mp3_compression',
        'neg_emphesis',
        'normalize',
        'pitch_shift',
        'polarity_inversion',
        'room_simulator',
        'seven_band_parameter'
    ] 
    base_path = f"D:/new_data/{keyword}/processed/{number}"
    _augmented = lambda x: f'{base_path}/augmentation/{x}'
    all_folders = list()
    real = f"{base_path}/cleaning/{real}"
    augmented = [_augmented(i) for i in augmentations]
    all_folders.append(real)
    all_folders.extend(augmented)
    files = [f'{i}/{j}' for i in all_folders for j in os.listdir(i)]
    return files

def duration_grouping(duration):
    ranges = [
        (0.5, '0.0 - 0.5'),
        (1.0, '0.5 - 1.0'),
        (1.5, '1.0 - 1.5'),
        (2.0, '1.5 - 2.0'),
        (2.5, '2.0 - 2.5'),
        (3.0, '2.5 - 3.0'),
        (3.5, '2.0 - 3.5'),
        (4.0, '3.5 - 4.0'),
        (4.5, '4.0 - 4.5'),
        (5.0, '4.5 - 5.0'),
    ]
    
    for lim, label in ranges:
        if duration <= lim:
            return label

def get_audio_info(audio_path):
    with wave.open(audio_path, 'rb') as wav_file:
        sample_rate = wav_file.getframerate()
        num_frames = wav_file.getnframes()
        duration = num_frames / sample_rate
        group = duration_grouping(duration)
    return sample_rate,duration,audio_path,group
