####python -m pip install --user numpy scipy matplotlib ipython jupyter pandas sympy nose

import sounddevice as sd
from scipy.io.wavfile import write
# import simpleaudio as sa
from playsound import playsound


def record(filename):    
    fs = 44100  # Sample rate
    seconds = 3  # Duration of recording

    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
    sd.wait()  # Wait until recording is finished
    write(filename, fs, myrecording)  # Save as WAV file 


def play(filename):
    # wave_obj = sa.WaveObject.from_wave_file(filename)
    # play_obj = wave_obj.play()
    # play_obj.wait_done()
    playsound(filename)

record('output.wav')
play('output.wav')