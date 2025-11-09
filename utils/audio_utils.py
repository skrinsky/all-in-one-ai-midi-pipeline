import librosa, soundfile as sf, numpy as np

def load_audio_mono(path, sr=44100):
    y, s = librosa.load(path, sr=sr, mono=True)
    return y, s

def write_audio(path, y, sr):
    sf.write(path, y, sr)

def rms(y):
    return np.sqrt(np.mean(np.square(y)))
