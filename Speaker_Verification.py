import numpy as np 
from pathlib import Path
from itertools import groupby
from tqdm import tqdm

from resemblyzer import preprocess_wav, VoiceEncoder

encoder = VoiceEncoder()


class ASV():

    def __init__(self, threshold=0.5):

        self._voice_embeddings = []
        self._speaker_embeddings = []
        self.theshold = threshold
        self.Verified = False

    
    def extract_features(self, voice_sample):

        voice_embedding = encoder.embed_utterance(voice_sample)

        # self._voice_embeddings.append(voice_embedding)

        return voice_embedding


    def register_speaker(self, wavs_list):
        '''
        adds a new speaker
        '''

        speaker_embed = encoder.embed_speaker(wavs_list)
        
        self._speaker_embeddings.append(speaker_embed)

        # return speaker_embed


    def compute_similarity(self, embed_a, embed_b): 

        sim = np.inner(embed_a, embed_b)

        return sim

    
    def verify_speaker(self, test_wav):
        '''
        verify a new test sample
        '''

        # compute the embedding of the test sample
        # test_embedd = encoder.embed_utterance(test_wav)
        test_embedd = self.extract_features(test_wav)

        # verify it against the registered speakers
        for spk in self._speaker_embeddings:

            sim_test_sample = self.compute_similarity(spk, test_embedd)

            if sim_test_sample > self.theshold:

                self.Verified = True

                break
            # else:
            #     self.Verified = False
        
        if self.Verified:
            print('Speaker is Verified')
        else:
            print("Speaker is Not Registered")

    def train(self, X, y):

        '''
        Used to train a speaker recognition model

        X: a numpy array containing training data of shape mxn where m is the number of samples and n is the length of one sample
        y: a numpy array or list containing true labels against training data of size mx1 where ith entry is the label of ith row in X 
        '''

        #TODO: implement a nearest neighbour algorithm. Train it on the voice dataset, already available in the repository.
        #  

        return X



if __name__ == "__main__":

    # 1. extract features from all the audio files
    # 2. conctenate all the features into one variable, lets say called X
    # 3. concatenate all the labels into one variable, lets say called y
    # 4. you may need to divide the dataset into train and test

    wav_fpaths = list(Path("audio_data", "librispeech_test-other").glob("**/*.flac"))
    # speaker_1_paths = list(Path("audio_data", "librispeech_test-other", "367").glob("*.flac"))

    # print(wav_fpaths)

    speaker_wavs = {speaker: list(map(preprocess_wav, wav_fpaths)) for speaker, wav_fpaths in
                groupby(tqdm(wav_fpaths, "Preprocessing wavs", len(wav_fpaths), unit="wavs"), 
                        lambda wav_fpath: wav_fpath.parent.stem)}

    # print(speaker_wavs)

    speaker1_wavs = speaker_wavs['367']
    # y = ['367'] * 10
    # y = np.array(y)

    # print(speaker1_wavs)
    # print(np.shape(speaker1_wavs))

    # # print(y)
    speaker2_wavs = speaker_wavs['533']

    # create the speaker verification class 
    asv = ASV(threshold=0.8)

    # # speaker1_embed = []
    # # for feat in speaker1_wavs:
    # #     embed = asv.extract_features(feat)

    # #     speaker1_embed.append(embed)

    # # speaker1_embed = np.array(speaker1_embed)

    # # print(speaker1_embed)

    # register the first new speakers
    asv.register_speaker(speaker1_wavs)

    # register the 2nd new speaker
    asv.register_speaker(speaker2_wavs)

    # print(np.shape(asv._speaker_embeddings[0]))

    # test a sample
    # test_sample = speaker_wavs['367'][2]
    test_sample = speaker_wavs['1688'][5]
    
    test_features = asv.extract_features(test_sample)
    
    sim = asv.compute_similarity(asv._speaker_embeddings[1], test_features)
    print(sim)

    asv.verify_speaker(test_sample)