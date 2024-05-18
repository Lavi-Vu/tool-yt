
from pytube import YouTube
import os
from pydub import AudioSegment
import whisper

# Function to download audio from YouTube
def download_audio(url, output_path="audio.mp3"):
    yt = YouTube(url)
    stream = yt.streams.filter(only_audio=True).first()
    stream.download(filename=output_path)

    # Convert to WAV format
    audio = AudioSegment.from_file(output_path)
    audio.export("audio.wav", format="wav")

# Function to transcribe audio using Whisper
def transcribe_audio(file_path="audio.wav"):
    model = whisper.load_model("base")
    result = model.transcribe(file_path)
    return result

# Example usage
youtube_url = "https://www.youtube.com/watch?v=U_pa4E23elQ&t=1s&pp=ygUJamFwYW4gbmV3"
# download_audio(youtube_url)
transcription = transcribe_audio("audio.wav")
# print(transcription)
for segment in transcription["segments"]:
    # print(f"Segment {segment['id']} ({segment['start']} - {segment['end']} seconds): {segment['text']}")
    print(f"{segment['text']}\t\t{len(segment['text'])}")

