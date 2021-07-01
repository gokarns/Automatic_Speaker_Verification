from playsound import playsound
import numpy as np
import simpleaudio as sa
import sounddevice as sd
from scipy.io.wavfile import write

fs = 44100  # Sample rate
seconds = 3  # Duration of recording

myrecording = sa.rec(int(seconds * fs), samplerate=fs, channels=2)
sa.wait()  # Wait until recording is finished
write('output.wav', fs, myrecording)  # Save as WAV file 