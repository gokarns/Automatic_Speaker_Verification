# Run the VAD on 10 ms of silence. The result should be False.
import webrtcvad
vad = webrtcvad.Vad(2)

sample_rate = 16000
frame_duration = 10  # ms
frame = b'\x00\x00' * int(sample_rate * frame_duration / 1000)
print('Contains speech: %s' % (vad.is_speech(frame, sample_rate)))