import numpy as np
import librosa

pad_or_trunc = lambda a,i : a[0:i] if len(a) > i else a if len(a) == i else np.pad(a,(0, (i-len(a))))


def rename(newname):
    def decorator(f):
        f.__name__ = newname
        return f
    return decorator

@rename('mel_spectogram')
def get_mel_spectogram(waveform, sr):
    # Compute the Mel spectogram
    S = librosa.feature.melspectrogram(y=waveform, sr=sr)
    S_dB = librosa.power_to_db(S, ref=np.max)
    return S_dB

@rename('mfcc')
def get_mfcc(waveform, sr, n_mfcc=13):
    # Compute the MFCCs
    mfccs = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=n_mfcc)
    return mfccs


feature_extractors = {
    'mfcc': get_mfcc,
    'mel_spectogram': get_mel_spectogram,
}