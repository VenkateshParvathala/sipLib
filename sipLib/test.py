import librosa
from specplots import plotSpectrogram

wav1, fs = librosa.load('/home/venkatesh/SE/Comparisons/clean/p257_001.wav', sr=None)
wav2, fs = librosa.load('/home/venkatesh/SE/Comparisons/clean/p257_002.wav', sr=None)
wav3, fs = librosa.load('/home/venkatesh/SE/Comparisons/clean/p257_003.wav', sr=None)

l = min(len(wav1), len(wav2), len(wav3))
wav1 = wav1[:l]; wav2=wav2[:l]; wav3=wav3[:l]

wavs = [wav1, wav2, wav3]
plotSpectrogram(wavs)
