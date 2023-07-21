import os
import glob
import re
import csv
from pydub import AudioSegment
import speech_recognition as sr

def transcribe_audio(audio_path):
    r = sr.Recognizer()

    with sr.AudioFile(audio_path) as source:
        audio_data = r.record(source)
        text = r.recognize_google(audio_data)
        return text

def get_next_file_number(directory):
    files = glob.glob(os.path.join(directory, 'Elon *.wav'))
    if not files:
        return 1
    else:
        highest_num = max(int(re.findall(r'\d+', file)[0]) for file in files)
        return highest_num + 1

def split_audio(file_path, directory):
    start_number = get_next_file_number(directory)
    audio = AudioSegment.from_wav(file_path)
    audio_length_in_ms = len(audio)
    one_min_in_ms = 60 * 1000  # 60 seconds * 1000 ms

    metadata = []

    for i, chunk_start in enumerate(range(0, audio_length_in_ms, one_min_in_ms), start=start_number):
        start_time = chunk_start
        end_time = chunk_start + one_min_in_ms
        if end_time > audio_length_in_ms:
            end_time = audio_length_in_ms  # Ensuring last chunk has remaining duration

        chunk = audio[start_time:end_time]

        chunk_name = os.path.join(directory, f"Elon {str(i).zfill(3)}.wav")
        chunk.export(chunk_name, format="wav")

        transcript = transcribe_audio(chunk_name)

        # Append the filename and transcript to metadata
        metadata.append([os.path.basename(chunk_name), transcript])

    # Write the metadata to a CSV file
    with open(os.path.join(directory, 'metadata.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter='|')
        writer.writerows(metadata)

def combine_audio_files(input_folder):
    combined_audio = AudioSegment.empty()

    # Iterate through all the files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".wav"):
            input_file = os.path.join(input_folder, filename)

            # Load the input audio using pydub
            audio = AudioSegment.from_file(input_file)

            # Append the audio to the combined audio
            combined_audio += audio

    return combined_audio

def split_audio_into_chunks(audio, chunk_duration):
    # Create an output directory
    output_directory = 'chunks'
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Calculate the total duration of the audio in milliseconds
    total_duration = len(audio)

    # Split the audio into one-minute chunks
    chunk_index = 1
    start_time = 0

    while start_time + chunk_duration * 1000 <= total_duration:
        end_time = start_time + chunk_duration * 1000

        # Generate the output filename
        output_file = os.path.join(output_directory, f"chunk_{chunk_index}.wav")

        # Extract the chunk from the audio and export as a single audio file
        chunk = audio[start_time:end_time]
        chunk.export(output_file, format="wav")

        # Update the start time and chunk index for the next chunk
        start_time = end_time
        chunk_index += 1


if __name__ == "__main__":
    input_folder = input("Enter the path to the input folder: ")
    output_directory = os.path.join(input_folder, 'chunks')
    chunk_duration = 60  # in seconds

    # Combine all the audio files in the input folder
    combined_audio = combine_audio_files(input_folder)

    # Split the combined audio into one-minute chunks
    split_audio_into_chunks(combined_audio, chunk_duration)
