import numpy as np
import simpleaudio as sa


filename = 'audios/firetruck1.wav'

# first sample
# wave_obj = sa.WaveObject.from_wave_file(filename)
# play_obj = wave_obj.play()
# play_obj.wait_done()  # Wait until sound has finished playing


# # second sample
# frequency = 440  # Our played note will be 440 Hz
# fs = 44100  # 44100 samples per second
# seconds = 3  # Note duration of 3 seconds

# # Generate array with seconds*sample_rate steps, ranging between 0 and seconds
# t = np.linspace(0, seconds, seconds * fs, False)

# # Generate a 440 Hz sine wave
# note = np.sin(frequency * t * 2 * np.pi)

# # Ensure that highest value is in 16-bit range
# audio = note * (2**15 - 1) / np.max(np.abs(note))
# # Convert to 16-bit data
# audio = audio.astype(np.int16)

# # Start playback
# play_obj = sa.play_buffer(audio, 1, 2, fs)

# # Wait for playback to finish before exiting
# play_obj.wait_done()


################### pyaudio playing sample ############
# import pyaudio
# import wave

# # filename = 'myfile.wav'

# # Set chunk size of 1024 samples per data frame
# chunk = 1024 

# # Open the sound file 
# wf = wave.open(filename, 'rb')

# # Create an interface to PortAudio
# p = pyaudio.PyAudio()

# # Open a .Stream object to write the WAV file to
# # 'output = True' indicates that the sound will be played rather than recorded
# stream = p.open(format = p.get_format_from_width(wf.getsampwidth()),
#                 channels = wf.getnchannels(),
#                 rate = wf.getframerate(),
#                 output = True)

# # Read data in chunks
# data = wf.readframes(chunk)

# # Play the sound by writing the audio data to the stream
# while data != '':
#     stream.write(data)
#     data = wf.readframes(chunk)

# # Close and terminate the stream
# stream.close()
# p.terminate()

 ####################    Sounddevice recording sample  ###################
import sounddevice as sd
from scipy.io.wavfile import write

fs = 44100  # Sample rate
seconds = 3  # Duration of recording

myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
sd.wait()  # Wait until recording is finished
write('audios/recording_sounddevice.wav', fs, myrecording)  # Save as WAV file


######################   pyaudio recording sample   #################

import pyaudio
import wave

chunk = 1024  # Record in chunks of 1024 samples
sample_format = pyaudio.paInt16  # 16 bits per sample
channels = 2
fs = 44100  # Record at 44100 samples per second
seconds = 3
filename = "audios/recording_pyaudio.wav"

p = pyaudio.PyAudio()  # Create an interface to PortAudio

print('Recording')

stream = p.open(format=sample_format,
                channels=channels,
                rate=fs,
                frames_per_buffer=chunk,
                input=True)

frames = []  # Initialize array to store frames

# Store data in chunks for 3 seconds
for i in range(0, int(fs / chunk * seconds)):
    data = stream.read(chunk)
    frames.append(data)

# Stop and close the stream 
stream.stop_stream()
stream.close()
# Terminate the PortAudio interface
p.terminate()

print('Finished recording')

# Save the recorded data as a WAV file
wf = wave.open(filename, 'wb')
wf.setnchannels(channels)
wf.setsampwidth(p.get_sample_size(sample_format))
wf.setframerate(fs)
wf.writeframes(b''.join(frames))
wf.close()