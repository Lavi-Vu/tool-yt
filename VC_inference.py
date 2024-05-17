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
import librosa
import webbrowser
from pydub import AudioSegment
import re
import nltk
nltk.download('punkt')

from text import text_to_sequence, _clean_text
device = "cuda:0" if torch.cuda.is_available() else "cpu"
import logging
import atexit

# Suppress PIL and urllib3 warnings
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("markdown_it").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)

# Language marks for TTS processing
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
        match = re.search(r'[。、.!?・・・\n]', chunk)
        if match:
            end_idx = match.end()
            chunk = text[start_idx:start_idx + end_idx]
        chunks.append(chunk.strip())
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

def create_tts_fn(model_dir, hps, speaker_ids):
    del model  # Release model after inference
    torch.cuda.empty_cache()  # Clear GPU memory
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
            
            
        
        return "Success", (hps.data.sampling_rate, audio)

    return tts_fn

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="./G_latest.pth", help="directory to your fine-tuned model")
    parser.add_argument("--config_dir", default="./finetune_speaker.json", help="directory to your model config file")
    parser.add_argument("--share", default=False, help="make link public (used in colab)")

    args = parser.parse_args()
    hps = utils.get_hparams_from_file(args.config_dir)

    speaker_ids = hps.speakers
    tts_fn = create_tts_fn(args.model_dir, hps, speaker_ids)

    app = gr.Blocks()
    with app:
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
                    text_output = gr.Textbox(label="Message")
                    audio_output = gr.Audio(label="Output Audio", elem_id="tts-audio")
                    btn = gr.Button("Generate!")
                    btn.click(tts_fn, inputs=[textbox, char_dropdown, language_dropdown, duration_slider], outputs=[text_output, audio_output])
    
    webbrowser.open("http://127.0.0.1:7860")
    app.launch(share=args.share)
