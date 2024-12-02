import os
from typing import Any
import numpy as np
import gradio as gr
import utils
from models import SynthesizerTrn

import torch
from torch import no_grad, LongTensor

import webbrowser
# import logging
import soundfile as sf

from _utils.text_process import *
from _utils.youtube import *
from _utils.video_process import *
import edge_tts
import platform

device = "cuda:0" if torch.cuda.is_available() else "cpu"

language_marks = {
    "Japanese": "[JA]",
}
lang = ['Japanese', 'English']

async def amain(text, voice) -> None:
    """Main function"""
    OUTPUT_FILE = "./create_video_demo/audio.wav"
    WEBVTT_FILE = "./create_video_demo/output.srt"
    
    communicate = edge_tts.Communicate(text, voice)
    submaker = edge_tts.SubMaker()
    with open(OUTPUT_FILE, "wb") as file:
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                file.write(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                submaker.create_sub((chunk["offset"], chunk["duration"]), chunk["text"])

    with open(WEBVTT_FILE, "w", encoding="utf-8") as file:
        file.write(submaker.generate_subs())
        
    # Modify subtitle for Windows 
    if(platform.system() == 'Windows'):
        with open(WEBVTT_FILE, "w", encoding="utf-8") as file:
            file.write(submaker.generate_subs())
        with open(WEBVTT_FILE, "r", encoding="utf-8") as file:
            lines = file.readlines()
        with open(WEBVTT_FILE, "w", encoding="utf-8") as file:
            for line in lines:
                if "-->" in line:
                    file.write(line.strip() + " ")
                else:
                    file.write(line)

def tts_fn(model_name, text, speaker, language, speed):
    # config_dir = model_list[model_name] + 'config.json'
    # model_dir = model_list[model_name] + 'G_latest.pth'
    # try:
    #     hps = utils.get_hparams_from_file(config_dir) 
    #     speaker_ids = hps.speakers
    #     print(config_dir)
    # except Exception as e :
    #     return f"Error: {str(e)}"
    # if (model_name != 'TTS'):
    #     with no_grad():
    #         model = SynthesizerTrn(
    #             len(hps.symbols),
    #             hps.data.filter_length // 2 + 1,
    #             hps.train.segment_size // hps.data.hop_length,
    #             n_speakers=hps.data.n_speakers,
    #             **hps.model).to(device).eval()
            
    #         _ = utils.load_checkpoint(model_dir, model, None)

    #         speaker_id = speaker_ids[speaker]
    #         chunks = split_text_by_length_or_character(text)
    #         audio = np.array([], dtype=np.float32)
    #         srt_content = []
    #         current_time = 0.0  # Initialize current time

    #         for idx, chunk in enumerate(chunks):
    #             chunk = remove_empty_lines(chunk)
    #             if len(chunk) > 0:
    #                 sub = chunk
    #                 if len(sub) > 30:
    #                     space_index = sub.rfind(' ', 0, 30)
    #                     if space_index != -1:
    #                         # Insert newline character at the space_index
    #                         sub = sub[:space_index] + '\n' + sub[space_index:]
    #                     else:
    #                         # If no space is found, just insert newline at position 30
    #                         sub = sub[:30] + '\n' + sub[30:]
                    
    #                 chunk = language_marks[language] + chunk + language_marks[language]
    #                 stn_tst = get_text(chunk, hps, False)
    #                 x_tst = stn_tst.unsqueeze(0).to(device)
    #                 x_tst_lengths = LongTensor([stn_tst.size(0)]).to(device)
    #                 sid = LongTensor([speaker_id]).to(device)
                    
    #                 audio_from_text = model.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8,
    #                                                 length_scale=1 / speed)[0][0, 0].data.cpu().float().numpy()
                    
    #                 start_time = current_time
    #                 duration = len(audio_from_text) / hps.data.sampling_rate
    #                 end_time = start_time + duration
    #                 current_time = end_time

    #                 srt_content.append(format_srt_entry(idx + 1, start_time, end_time, sub))
                    
    #                 audio = np.concatenate((audio, audio_from_text))
                    
    #                 del stn_tst, x_tst, x_tst_lengths, sid, audio_from_text  # Release temporary variables
            
    #         del model  # Release model after inference
    #         torch.cuda.empty_cache()  # Clear GPU memory
    #     if os.path.exists('create_video_demo/audio.wav'):
    #         os.remove('create_video_demo/audio.wav')
    #         sf.write('create_video_demo/audio.wav', audio, hps.data.sampling_rate)
    #     else:
    #         sf.write('create_video_demo/audio.wav', audio, hps.data.sampling_rate)
    #     save_srt_file("create_video_demo/output.srt", srt_content)
    #     return "Success", (hps.data.sampling_rate, audio)
    
    # elif(model_name == 'TTS'):
        import asyncio
        TEXT = text
        VOICE = speaker
        asyncio.run(amain(TEXT, speaker))
        audio,  samplerate = sf.read('./create_video_demo/audio.wav')
        return "Success", (samplerate, audio)



    
# # Function to transcribe audio using Whisper library
# def transcribe_audio_with_whisper(audio_path, model_type, language):
#     try:
#         model = WhisperModel(model_type, device="cuda", compute_type="int8")
#         if language == "auto":
#             result = model.transcribe(audio_path, beam_size=5)
#         else:
#             text = str()
#             segments, _ = model.transcribe(audio_path, beam_size=5, language=language)
#             for segment in segments:
#                 text += segment.text
#         del model
#         torch.cuda.empty_cache()
#         gc.collect()
#         return text
#     except Exception as e:
#         print(f"Error transcribing audio with Whisper library: {e}")
#         return None


# # Function to transcribe audio and add punctuation
# def transcribe_audio(youtube_url, model_type, language):
#     audio_path = download_audio(youtube_url)
#     if audio_path:
#         if model_type != 'turbo':
#             transcription = transcribe_audio_with_whisper(audio_path, model_type, language)
#             os.remove(audio_path)  # Remove the audio file after transcription
#             if transcription:
#                 return add_punctuation(transcription)
#             else:
#                 return "Transcription failed"
#         elif model_type == 'turbo':
#             import whisper

#             model = whisper.load_model("turbo")
#             transcription = model.transcribe(audio_path)
#             print(transcription["text"])
#             del model
#             torch.cuda.empty_cache()
#             gc.collect()
#             if transcription and language != 'en':
#                 return add_punctuation(transcription["text"])
#             elif transcription and language == 'en':
#                 return transcription["text"]
#             else:
#                 return "Transcription failed"
#     else:
#         return "Audio download failed"

def get_model_params(model_name):
    config_dir = model_list[model_name] + 'config.json'
    model_dir = model_list[model_name] + 'G_latest.pth'
    print(config_dir)
    try:
        hps = utils.get_hparams_from_file(config_dir) 
        speaker_ids = hps.speakers
        print(hps)
        print(config_dir)
        print(speaker_ids)
        return gr.update(choices=list(speaker_ids.keys()), value=None)

    except Exception as e :
        return gr.update(choices=[], value=None)
    
def toggle_file_upload(show_files):
    # Show file upload components if checkbox is checked
    if show_files:
        return gr.update(visible=True), gr.update(visible=True)
    else:
        return gr.update(visible=False), gr.update(visible=False)


model_list = {'VBEE': './MODEL_VBEE/',
        'OPENAI' : './MODEL_OPENAI/',
        'TTS' : './TTS/',
        }
if __name__ == "__main__":
    with gr.Blocks(delete_cache=(3600, 3600)) as demo:
        with gr.Tab("Text-to-Speech"):
            with gr.Row():
                with gr.Column():
                    textbox = gr.TextArea(label="Text",
                                          placeholder="Type your sentence here",elem_id="tts-input")
                    model_name = gr.Dropdown(choices=list(model_list.keys()),label="Select Model", interactive=True, value=None)
                    char_dropdown = gr.Dropdown(choices=[], label='character', interactive=True)
                    language_dropdown = gr.Dropdown(choices=lang, value=lang[0], label='language')
                    duration_slider = gr.Slider(minimum=0.1, maximum=3, value=1, step=0.1, label='Speed')
                    
                    model_name.change(fn=get_model_params, inputs=model_name, outputs=[char_dropdown])
                with gr.Column():
                    text_output = gr.Textbox(label="Message")
                    audio_output = gr.Audio(label="Output Audio", elem_id="tts-audio")
                    btn = gr.Button("Generate!")
                    btn.click(tts_fn, inputs=[model_name,textbox, char_dropdown, language_dropdown, duration_slider], outputs=[text_output, audio_output])
        # with gr.Tab("TOOL lay SUB"):
        #     gr.Markdown("# YouTube Video Transcriber")
        #     gr.Markdown("Enter the URL of a YouTube video to download, extract audio, and transcribe using Whisper.")
            
        #     with gr.Row():
        #         with gr.Column():
        #             url_input = gr.Textbox(label="YouTube Video URL", placeholder="Enter YouTube video URL here...")
        #             model_select = gr.Dropdown(label="Select Model Type", choices=["base", "medium", "turbo"], value="turbo")
        #             language_select = gr.Dropdown(label="Select Language", choices=["ko","ja","en","auto"], value="ja")
        #             transcribe_button = gr.Button("Transcribe")
        #         with gr.Column():
        #             transcription_output = gr.Textbox(label="Transcription")
            
        #     transcribe_button.click(fn=transcribe_audio, inputs=[url_input, model_select, language_select], outputs=transcription_output)
        with gr.Tab("TOOL RENDER VIDEO"):
            # Define interface components
            input_video = gr.File(label="Input Video (.mp4)", type="filepath")
            checkbox = gr.Checkbox(label="Them sub va audio", value=False)
            subtitle_file = gr.File(label="Subtitle File (.srt)", type="filepath", visible=False)
            audio_file = gr.File(label="Audio File (.mp3 or .wav)", type="filepath", visible=False)
            output_video = gr.Video(label="Output Video (.mp4)")

            # Update visibility of subtitle and audio file fields when checkbox changes
            checkbox.change(
                toggle_file_upload,
                inputs=checkbox,
                outputs=[subtitle_file, audio_file]
            )

            # Create the interface for processing
            iface = gr.Interface(
                fn=process,
                inputs=[input_video, checkbox, subtitle_file, audio_file],
                outputs=output_video,
                title="Add Audio and Subtitles with GPU Acceleration",
                description="Upload a video, an .srt file, and an audio file to add subtitles and replace the audio using GPU acceleration.",
            )
            
        # with gr.Tab("TOOL CHUYEN GIONG"):
        #     inputs = [
        #         gr.Audio(type="filepath", label="Source Audio"),
        #         gr.Audio(type="filepath", label="Reference Audio"),
        #         gr.Slider(minimum=1, maximum=200, value=10, step=1, label="Diffusion Steps", info="10 by default, 50~100 for best quality"),
        #         gr.Slider(minimum=0.5, maximum=2.0, step=0.1, value=1.0, label="Length Adjust", info="<1.0 for speed-up speech, >1.0 for slow-down speech"),
        #         gr.Slider(minimum=0.0, maximum=1.0, step=0.1, value=0.7, label="Inference CFG Rate", info="has subtle influence"),
        #         gr.Slider(minimum=1, maximum=3, step=1, value=1, label="N Quantizers", info="the less quantizer used, the less prosody of source audio is preserved"),
        #     ]

        #     outputs = [gr.Audio(label="Stream Output Audio / 流式输出", streaming=True, format='mp3'),
        #        gr.Audio(label="Full Output Audio / 完整输出", streaming=False, format='wav')]
        #     gr.Interface(fn=voice_conversion,
        #          inputs=inputs,
        #          outputs=outputs,
        #          title="Voice Conversion",
        #          )
    webbrowser.open("http://127.0.0.1:7861")
    demo.launch(server_port=7861, share=True)