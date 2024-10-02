import os
import numpy as np
import torch
from torch import no_grad, LongTensor
import argparse
import commons
from mel_processing import spectrogram_torch
import utils
from models import SynthesizerTrn
import gradio as gr
import re
import docx
from text import text_to_sequence, _clean_text
from faster_whisper import WhisperModel
import gc 
import yt_dlp
import webbrowser
device = "cuda:0" if torch.cuda.is_available() else "cpu"
import logging
import subprocess
import pysrt
import ffmpeg
import soundfile as sf
from datetime import timedelta

logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("markdown_it").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)

language_marks = {
    "Japanese": "[JA]",
    "Korea": "[KO]",

}
# lang = ['Japanese', 'Korea']
lang = ['Korea']

# def split_text_by_length_or_character(text):
#     max_chunk_length = 40
#     chunks = []
#     start_idx = 0
#     while start_idx < len(text):
#         end_idx = min(start_idx + max_chunk_length, len(text))
#         chunk = text[start_idx:end_idx]
#         # Tìm vị trí của dấu chấm câu hoặc dấu xuống dòng
#         # match = re.search(r'[。、.!?・・・\n]', chunk)
#         # if match:
#         #     end_idx = match.end()  # Lấy vị trí kết thúc của câu
#         #     chunk = text[start_idx:start_idx + end_idx]
#         chunks.append(chunk.strip())  # Thêm phần chunk vào danh sách chunks
#         start_idx += len(chunk)
#     return chunks
def split_text_by_length_or_character(text):
    lines = text.split('\n')
    processed_lines = []
    current_line = ""

    for line in lines:
        if current_line:
            current_line += line
        else:
            current_line = line
        
        while len(current_line) >= 40:
            processed_lines.append(current_line[:40])
            current_line = current_line[40:]
    
    if current_line:  # Append any remaining text as the last line
        processed_lines.append(current_line)

    return processed_lines


def get_text(text, hps, is_symbol):
    text_norm = text_to_sequence(text, hps.symbols, [] if is_symbol else hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = LongTensor(text_norm)
    return text_norm

def remove_empty_lines(text):
    lines = text.splitlines()
    cleaned_lines = [line for line in lines if line.strip()]
    cleaned_text = '\n'.join(cleaned_lines)
    return cleaned_text

def read_txt_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content

def read_docx_file(file_path):
    doc = docx.Document(file_path)
    content = []
    for paragraph in doc.paragraphs:
        content.append(paragraph.text)
    return '\n'.join(content)

# Gradio Interface
def file_or_text_reader(text, file):
    if file:
        # If the input is a file, determine its format and read its content
        if file.name.endswith('.txt'):
            content = read_txt_file(file.name)
        elif file.name.endswith('.docx'):
            content = read_docx_file(file.name)
        else:
            content = "Unsupported file format. Please upload a .txt or .docx file."
    if text:
        # If the input is text, return it directly
        content = text
    return content

def format_srt_entry(index, start_time, end_time, text):
    start = format_time(start_time)
    end = format_time(end_time)
    return f"{index}\n{start} --> {end}\n{text}\n"

def format_time(seconds):
    milliseconds = int((seconds % 1) * 1000)
    seconds = int(seconds)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

def save_srt_file(filename, content):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("\n".join(content))
def create_tts_fn(model_dir, hps, speaker_ids):
    def tts_fn(text, speaker, language, speed):
        with no_grad():
            model = SynthesizerTrn(
                len(hps.symbols),
                hps.data.filter_length // 2 + 1,
                hps.train.segment_size // hps.data.hop_length,
                n_speakers=hps.data.n_speakers,
                **hps.model).to(device).eval()
            
            _ = utils.load_checkpoint(model_dir, model, None)

            speaker_id = speaker_ids[speaker]
            chunks = split_text_by_length_or_character(text)
            audio = np.array([], dtype=np.float32)
            srt_content = []
            current_time = 0.0  # Initialize current time

            for idx, chunk in enumerate(chunks):
                chunk = remove_empty_lines(chunk)
                if len(chunk) > 0:
                    sub = chunk
                    if len(sub) > 30:
                        space_index = sub.rfind(' ', 0, 30)
                        if space_index != -1:
                            # Insert newline character at the space_index
                            sub = sub[:space_index] + '\n' + sub[space_index:]
                        else:
                            # If no space is found, just insert newline at position 30
                            sub = sub[:30] + '\n' + sub[30:]
                    stn_tst = get_text(chunk, hps, False)
                    x_tst = stn_tst.unsqueeze(0).to(device)
                    x_tst_lengths = LongTensor([stn_tst.size(0)]).to(device)
                    sid = LongTensor([speaker_id]).to(device)
                    
                    audio_from_text = model.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.6,
                                                  length_scale=0.8 / speed)[0][0, 0].data.cpu().float().numpy()
                    
                    start_time = current_time
                    duration = len(audio_from_text) / hps.data.sampling_rate
                    end_time = start_time + duration
                    current_time = end_time

                    srt_content.append(format_srt_entry(idx + 1, start_time, end_time, sub))
                    
                    audio = np.concatenate((audio, audio_from_text))
                    
                    del stn_tst, x_tst, x_tst_lengths, sid, audio_from_text  # Release temporary variables
            
            del model  # Release model after inference
            torch.cuda.empty_cache()  # Clear GPU memory
        if os.path.exists('create_video_demo/audio.wav'):
            os.remove('create_video_demo/audio.wav')
            sf.write('create_video_demo/audio.wav', audio, hps.data.sampling_rate)
        else:
            sf.write('create_video_demo/audio.wav', audio, hps.data.sampling_rate)
        save_srt_file("create_video_demo/output.srt", srt_content)
        return "Success", (hps.data.sampling_rate, audio)

    return tts_fn
# Function to download YouTube video
def download_audio(url):
    try:
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': 'downloaded_audio.%(ext)s',
            'quiet': True
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=True)
            audio_file = ydl.prepare_filename(info_dict).rsplit('.', 1)[0] + ".mp3"
        return audio_file
    except Exception as e:
        print(f"Error downloading audio: {e}")
        return None

# Function to transcribe audio using Whisper library
def transcribe_audio_with_whisper(audio_path, model_type, language):
    try:
        model = WhisperModel(model_type, device="cuda", compute_type="int8_float16")
        if language == "auto":
            result = model.transcribe(audio_path, beam_size=5)
        else:
            text = str()
            segments, _ = model.transcribe(audio_path, beam_size=5, language=language)
            for segment in segments:
                text += segment.text
        del model
        torch.cuda.empty_cache()
        gc.collect()
        return text
    except Exception as e:
        print(f"Error transcribing audio with Whisper library: {e}")
        return None
def add_punctuation(text):
    punctuated_text = ""
    count = 0
    for char in text:
        count += 1
        punctuated_text += char
        if char == '。' or count % 40 ==0:  # Sentence-ending period in Japanese (full stop)
            punctuated_text += '\n'
            count = 0
    
    punctuated_text = punctuated_text.strip()
    if not punctuated_text.endswith('\n'):
        punctuated_text += '\n'
    return punctuated_text
# Function to transcribe audio and add punctuation
def transcribe_audio(youtube_url, model_type, language):
    audio_path = download_audio(youtube_url)
    if audio_path:
        transcription = transcribe_audio_with_whisper(audio_path, model_type, language)
        os.remove(audio_path)  # Remove the audio file after transcription
        if transcription:
            return add_punctuation(transcription)
        else:
            return "Transcription failed"
    else:
        return "Audio download failed"


def add_audio_and_subtitles_to_video(input_video_path, subtitle_file_path, audio_file_path, output_video_path):
    subs = pysrt.open(subtitle_file_path)
    last_sub_end_time = subs[-1].end.to_time() if subs else timedelta(seconds=0)

    # Convert last_sub_end_time to timedelta for total_seconds calculation
    last_sub_end_time = timedelta(hours=last_sub_end_time.hour,
                                  minutes=last_sub_end_time.minute,
                                  seconds=last_sub_end_time.second,
                                  microseconds=last_sub_end_time.microsecond)

    # Calculate the total duration needed in seconds
    subtitle_end_time_sec = last_sub_end_time.total_seconds()

    # Construct the ffmpeg command
    input_video = ffmpeg.input(input_video_path, stream_loop=-1)
    input_audio = ffmpeg.input(audio_file_path)

    if os.path.exists(output_video_path):
        os.remove(output_video_path)

    # Apply subtitles with black background
    video_with_subtitles = input_video.filter('subtitles', subtitle_file_path, force_style='OutlineColour=&H00000000,BorderStyle=3,Outline=1,FontSize=22')

    # Apply blur and black background to the subtitles
    video_with_blur_background = video_with_subtitles.filter('drawtext',fontsize=3, fontcolor='white', text='', box=1, boxcolor='black@0.1', boxborderw=5, x='(w-text_w)/2', y='h-30')

    # Trim the video to the end time of the subtitles
    trimmed_video = ffmpeg.trim(video_with_blur_background, duration=subtitle_end_time_sec)
    trimmed_video = ffmpeg.setpts(trimmed_video, 'PTS-STARTPTS')

    # Combine video with new audio, trimming audio to match the video length
    output = ffmpeg.output(trimmed_video, input_audio, output_video_path,
                           vcodec='h264_nvenc', acodec='aac',
                           video_bitrate='5M', strict='experimental')

    ffmpeg.run(output)
    return output_video_path

def process(video):
    input_video_path = video.name
    subtitle_file_path = 'create_video_demo/output.srt'
    audio_file_path = 'create_video_demo/audio.wav'
    output_video_path = "create_video_demo/output_gradio_for_demo.mp4"

    # Add audio and subtitles to the video
    output_path = add_audio_and_subtitles_to_video(input_video_path, subtitle_file_path, audio_file_path, output_video_path)
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="./G_latest.pth", help="directory to your fine-tuned model")
    parser.add_argument("--config_dir", default="./finetune_speaker.json", help="directory to your model config file")

    args = parser.parse_args()
    hps = utils.get_hparams_from_file(args.config_dir)

    speaker_ids = hps.speakers
    tts_fn = create_tts_fn(args.model_dir, hps, speaker_ids)

    with gr.Blocks(delete_cache=(3600, 3600)) as demo:
        with gr.Tab("Text-to-Speech"):
            with gr.Row():
                with gr.Column():
                    textbox = gr.TextArea(label="Text",
                                          placeholder="Type your sentence here",
                                            elem_id="tts-input")
                    char_dropdown = gr.Dropdown(choices=speaker_ids.keys(), value=list(speaker_ids.keys())[0], label='character')
                    # char_dropdown = gr.Dropdown(choices=speaker_ids, value=speaker_ids[0], label='character')
                    language_dropdown = gr.Dropdown(choices=lang, value=lang[0], label='language')
                    duration_slider = gr.Slider(minimum=0.1, maximum=5, value=1, step=0.1, label='Speed')
                with gr.Column():
                    text_output = gr.Textbox(label="Message")
                    audio_output = gr.Audio(label="Output Audio", elem_id="tts-audio")
                    btn = gr.Button("Generate!")
                    btn.click(tts_fn, inputs=[textbox, char_dropdown, language_dropdown, duration_slider], outputs=[text_output, audio_output])
        with gr.Tab("TOOL lay SUB"):
            gr.Markdown("# YouTube Video Transcriber")
            gr.Markdown("Enter the URL of a YouTube video to download, extract audio, and transcribe using Whisper.")
            
            with gr.Row():
                with gr.Column():
                    url_input = gr.Textbox(label="YouTube Video URL", placeholder="Enter YouTube video URL here...")
                    model_select = gr.Dropdown(label="Select Model Type", choices=["base", "medium"], value="base")
                    language_select = gr.Dropdown(label="Select Language", choices=["ko","ja","auto"], value="ja")
                    transcribe_button = gr.Button("Transcribe")
                with gr.Column():
                    transcription_output = gr.Textbox(label="Transcription")
            
            transcribe_button.click(fn=transcribe_audio, inputs=[url_input, model_select, language_select], outputs=transcription_output)
        with gr.Tab("TOOL RENDER VIDEO"):
            iface = gr.Interface(
                fn=process,
                inputs=[
                    gr.File(label="Input Video (.mp4)", type="filepath"),
                    # gr.File(label="Subtitle File (.srt)", type="filepath"),
                    # gr.File(label="Audio File (.mp3 or .wav)", type="filepath")
                ],
                outputs=gr.Video(label="Output Video (.mp4)"),
                title="Add Audio and Subtitles with GPU Acceleration",
                description="Upload a video, an .srt file, and an audio file to add subtitles and replace the audio using GPU acceleration."
            )
        # with gr.Tab("TOOL AIO"):        
        #     with gr.Row():
        #         with gr.Column():
        #             video_input = gr.File(label="Upload Video", file_types=["video"])
        #             text_input = gr.Textbox(label="Enter Youtube link")
        #             model_options= gr.Dropdown(choices=["base", "medium"], label='model')
        #             char_dropdown = gr.Dropdown(choices=speaker_ids.keys(), label='character')
        #             language_dropdown = gr.Dropdown(choices=lang, label='language')
        #             duration_slider = gr.Slider(minimum=0.1, maximum=5, value=1, step=0.1, label='Speed')
        #         with gr.Column():
        #             video_output = gr.Video(label="Preview Video")
        #             transcript_output = gr.Textbox(label="Transcript")
        #             audio_output = gr.Audio(label="Output Audio", elem_id="tts-audio")

        #     submit_button = gr.Button("Submit")
        #     submit_button.click(
        #         process_video, 
        #         inputs=[video_input, text_input, model_options, char_dropdown, language_dropdown, duration_slider], 
        #         outputs=[video_output, transcript_output, audio_output]
        #     )   
    webbrowser.open("http://127.0.0.1:7860")
    demo.launch(server_port=7860)