import os
import json
import torchaudio
raw_audio_dir = "/home/lavi/Downloads/data/vbee/data_croped/shiori/"
denoise_audio_dir = "/home/lavi/Downloads/data/vbee/denoise/shiori/"
filelist = list(os.walk(raw_audio_dir))[0][2]
# 2023/4/21: Get the target sampling rate
with open("./configs/finetune_speaker.json", 'r', encoding='utf-8') as f:
    hps = json.load(f)
target_sr = hps['data']['sampling_rate']
for file in filelist:
    if file.endswith(".wav"):
        os.system(f"demucs --two-stems=vocals {raw_audio_dir}{file}")
for file in filelist:
    print(file)
    file = file.replace(".wav", "")
    wav, sr = torchaudio.load(f"{raw_audio_dir}/{file}", frame_offset=0, num_frames=-1, normalize=True,
                              channels_first=True)
    # merge two channels into one
    wav = wav.mean(dim=0).unsqueeze(0)
    if sr != target_sr:
        wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(wav)
    torchaudio.save(denoise_audio_dir + file + ".wav", wav, target_sr, channels_first=True)