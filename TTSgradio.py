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
import ffmpeg
import whisper
import gc 
import yt_dlp
device = "cuda:0" if torch.cuda.is_available() else "cpu"
import logging


logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("markdown_it").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)

language_marks = {
    "Japanese": "[JA]",
}
lang = ['Japanese']
def split_text_by_length_or_character(text):
    max_chunk_length = 50
    chunks = []
    start_idx = 0
    while start_idx < len(text):
        end_idx = min(start_idx + max_chunk_length, len(text))
        chunk = text[start_idx:end_idx]
        # Tìm vị trí của dấu chấm câu hoặc dấu xuống dòng
        match = re.search(r'[。、.!?・・・\n]', chunk)
        if match:
            end_idx = match.end()  # Lấy vị trí kết thúc của câu
            chunk = text[start_idx:start_idx + end_idx]
        chunks.append(chunk.strip())  # Thêm phần chunk vào danh sách chunks
        start_idx += len(chunk)
    return chunks

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

            for chunk in chunks:
                chunk = remove_empty_lines(chunk)
                if len(chunk) > 0:
                    chunk = language_marks[language] + chunk + language_marks[language]
                    stn_tst = get_text(chunk, hps, False)
                    x_tst = stn_tst.unsqueeze(0).to(device)
                    x_tst_lengths = LongTensor([stn_tst.size(0)]).to(device)
                    sid = LongTensor([speaker_id]).to(device)
                    audio_from_text = model.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8,
                                                  length_scale=1.0 / speed)[0][0, 0].data.cpu().float().numpy()
                    audio = np.concatenate((audio, audio_from_text))
                    del stn_tst, x_tst, x_tst_lengths, sid, audio_from_text  # Release temporary variables
            
            del model  # Release model after inference
            torch.cuda.empty_cache()  # Clear GPU memory
        
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
def transcribe_audio_with_whisper(audio_path, model_type):
    try:
        model = whisper.load_model(model_type)
        result = model.transcribe(audio_path)
        del model
        torch.cuda.empty_cache()
        gc.collect()
        return result["text"]
    except Exception as e:
        print(f"Error transcribing audio with Whisper library: {e}")
        return None

# Function to transcribe audio
def transcribe_audio(youtube_url, model_type):
    audio_path = download_audio(youtube_url)
    if audio_path:
        transcription = transcribe_audio_with_whisper(audio_path,model_type)
        os.remove(audio_path)
        return transcription if transcription else "Transcription failed"
    else:
        return "Audio download failed"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="./G_latest.pth", help="directory to your fine-tuned model")
    parser.add_argument("--config_dir", default="./finetune_speaker.json", help="directory to your model config file")

    args = parser.parse_args()
    hps = utils.get_hparams_from_file(args.config_dir)

    speaker_ids = hps.speakers
    tts_fn = create_tts_fn(args.model_dir, hps, speaker_ids)

    with gr.Blocks() as demo:
        with gr.Tab("Text-to-Speech"):
            with gr.Row():
                with gr.Column():
                    textbox = gr.TextArea(label="Text",
                                          placeholder="Type your sentence here",
                                          value="こんにちわ。", elem_id="tts-input")
                    char_dropdown = gr.Dropdown(choices=speaker_ids.keys(), value=list(speaker_ids.keys())[0], label='character')
                    language_dropdown = gr.Dropdown(choices=lang, value=lang[0], label='language')
                    duration_slider = gr.Slider(minimum=0.1, maximum=5, value=1, step=0.1, label='速度 Speed')
                with gr.Column():
                    model_options = ["base", "medium", ]
                    text_output = gr.Textbox(label="Message")
                    audio_output = gr.Audio(label="Output Audio", elem_id="tts-audio")
                    btn = gr.Button("Generate!")
                    btn.click(tts_fn, inputs=[textbox, char_dropdown, language_dropdown, duration_slider], outputs=[text_output, audio_output])
        with gr.Tab("TOOL lay SUB"):
           iface = gr.Interface(fn=transcribe_audio,  inputs=["text", gr.Dropdown(model_options, label="Select Model")], outputs="text", title="YouTube Video Transcription", description="Enter a YouTube URL to transcribe the audio using Whisper.")

                    

    demo.launch(share=True)
