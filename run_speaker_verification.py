import numpy as np 
from pathlib import Path
from itertools import groupby
from tqdm import tqdm
from sklearn import svm
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from resemblyzer import preprocess_wav, VoiceEncoder

from Speaker_Verification import ASV, create_X_Y


# get the dataset to run ASV on
# wav_fpaths = list(Path("/home", "hashim", "PHD", "vox1_dev_wav").glob("*id100[01-99]*/**/*.wav"))
wav_fpaths = list(Path("/home", "hashim", "PHD", "vox1_dev_wav").glob("*id1000[1-9]*/**/*.wav"))

wav_fpaths.sort()

# wav_fpaths_10 = wav_fpaths[0:10]

# print(Path("home", "hashim", "PHD", "vox1_dev_wav"))

print(wav_fpaths)

speaker_wavs = {speaker: list(map(preprocess_wav, wav_fpaths)) for speaker, wav_fpaths in
                groupby(tqdm(wav_fpaths, "Preprocessing wavs", len(wav_fpaths), unit="wavs"), 
                        lambda wav_fpath: wav_fpath.parent.parent.stem)}

# speaker_wavs = dict()
# for wav_fpath in wav_fpaths:

#     print(wav_fpath)

#     print(wav_fpath.parent.parent.stem)

print(speaker_wavs.keys())
# print(speaker_wavs['id10001'])
# print(speaker_wavs['id10002'])

# create an object of speaker verification class
asv = ASV(threshold=0.8)

##############   register some speakers    ############
speaker1_wavs = speaker_wavs['id10001']
speaker2_wavs = speaker_wavs['id10002']
speaker3_wavs = speaker_wavs['id10003']

print(np.shape(speaker1_wavs))
print(np.shape(speaker2_wavs))
print(np.shape(speaker3_wavs))

 # split into training and testing
speaker1_train, speaker1_test = train_test_split(speaker1_wavs, test_size=0.8)
speaker2_train, speaker2_test = train_test_split(speaker2_wavs, test_size=0.8)
speaker3_train, speaker3_test = train_test_split(speaker3_wavs, test_size=0.8)

print(np.shape(speaker1_train))
print(np.shape(speaker2_train))
print(np.shape(speaker3_train))

# register the speakers
asv.register_speaker(speaker1_train, id = 'id10001')
asv.register_speaker(speaker2_train, id = 'id10002')
asv.register_speaker(speaker3_train, id = 'id10003')

# verification part
test_sample = speaker3_test[5]
# speaker3_wavs = speaker_wavs['id10004']
# test_sample = speaker3_wavs[4]

# test_features = asv.extract_features(test_sample)
    
# sim = asv.compute_similarity(asv._speaker_embeddings['id10002'], test_features)
# print(sim)

asv.verify_speaker(test_sample, claimed_id='id10003')

##### speaker identification part #####

# X, y = create_X_Y(speaker_wavs)
    
# # print("X = ", X)

# # print("y = ", y)

# # embedding the voices in X
# X_embed = []
# for wav_processed in X:

#     # print(wav_processed)
#     X_embed.append(asv.extract_features(wav_processed))

# # print(X_embed)

# # split into training and testing
# X_train, X_test, y_train, y_test = train_test_split(X_embed, y, test_size=0.8)

# # print("train size = ", np.shape(X_train))
# # print("test size = ", np.shape(X_test))


# # train svm classifier
# asv.train_svm(X_train, y_train)

# y_pred = asv.lin_svm_clf.predict(X_test)

# print("Predicted labels = ", y_pred)

# print("accuracy = ", accuracy_score(y_test, y_pred))