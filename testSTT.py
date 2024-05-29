import whisper
from pytube import YouTube
import os
from pydub import AudioSegment
from janome.tokenizer import Tokenizer
import re
tokenizer = Tokenizer()
# Function to download YouTube audio
def download_youtube_audio(youtube_url, output_path="audio.mp3"):
    yt = YouTube(youtube_url)
    stream = yt.streams.filter(only_audio=True).first()
    stream.download(filename=output_path)
    return output_path

# Function to convert audio file to a format supported by Whisper (if needed)
def convert_audio(input_path, output_path="audio.wav"):
    audio = AudioSegment.from_file(input_path)
    audio.export(output_path, format="wav")
    return output_path

# Function to transcribe audio using Whisper
def transcribe_audio(file_path):
    model = whisper.load_model("base")
    result = model.transcribe(file_path,language="ja")
    print(result)
    return result["text"]


# Main function
def main(youtube_url):
    # Step 1: Download audio from YouTube
    audio_path = download_youtube_audio(youtube_url)

    # Step 2: Convert audio to WAV format (if needed)
    # wav_path = convert_audio(audio_path="/home/lavi/Documents/myprj/VITS-fast-fine-tuning/audio.mp3")
    wav_path = convert_audio(audio_path)
    
    # Step 3: Transcribe audio using Whisper
    # wav_path = "/home/lavi/Documents/myprj/VITS-fast-fine-tuning/audio.wav"
    result = []
    
    transcribed_text = transcribe_audio(wav_path)
    print(transcribed_text)
    # for segment in transcribed_text:
    #     result.append(segment['text'])
    #     print(segment['text'])
    # print(result)
    # Split the transcribed text into sentences
    # sentences = split_sentences(transcribed_text)

# # Add a period (。) at the end of each sentence
#     modified_sentences = [sentence if sentence.endswith("。") else sentence + "。" for sentence in sentences]

#     # Print the modified sentences
#     for sentence in modified_sentences:
#         print(sentence)


# Example usage
youtube_url = "https://www.youtube.com/watch?v=e7Y4CdzvhtI"
main(youtube_url)
