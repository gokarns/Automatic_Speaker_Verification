import numpy as np 
from pathlib import Path
from itertools import groupby
from tqdm import tqdm
from sklearn import svm
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from resemblyzer import preprocess_wav, VoiceEncoder

from sklearn.model_selection import train_test_split
from sklearn import neighbors, datasets

from scipy.io.wavfile import read, write 

from Speaker_Verification import ASV

asv = ASV(threshold=0.8)

audio_file = 'audios/firetruck1.wav'

audio_data = preprocess_wav(audio_file)


voice_embed = asv.extract_features(audio_data)

print(np.shape(voice_embed))
# print(audio_data)




