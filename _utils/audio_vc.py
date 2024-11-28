import torch
import torchaudio
from modules.commons import build_model, load_checkpoint, recursive_munch
import yaml
from hf_utils import load_custom_model_from_hf
import numpy as np
from pydub import AudioSegment
import librosa
import gc 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_models():    
    dit_checkpoint_path, dit_config_path = load_custom_model_from_hf("Plachta/Seed-VC",
                                                "DiT_step_298000_seed_uvit_facodec_small_wavenet_pruned.pth",
                                                "config_dit_mel_seed_facodec_small_wavenet.yml")

    config = yaml.safe_load(open(dit_config_path, 'r'))
    model_params = recursive_munch(config['model_params'])
    model = build_model(model_params, stage='DiT')
    hop_length = config['preprocess_params']['spect_params']['hop_length']
    sr = config['preprocess_params']['sr']

    # Load checkpoints
    model, _, _, _ = load_checkpoint(model, None, dit_checkpoint_path,
                                    load_only_params=True, ignore_modules=[], is_distributed=False)
    for key in model:
        model[key].eval()
        model[key].to(device)
    model.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)

    # Load additional modules
    from modules.campplus.DTDNN import CAMPPlus

    campplus_ckpt_path = load_custom_model_from_hf("funasr/campplus", "campplus_cn_common.bin", config_filename=None)
    campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
    campplus_model.load_state_dict(torch.load(campplus_ckpt_path, map_location="cpu"))
    campplus_model.eval()
    campplus_model.to(device)

    from modules.hifigan.generator import HiFTGenerator
    from modules.hifigan.f0_predictor import ConvRNNF0Predictor

    hift_checkpoint_path, hift_config_path = load_custom_model_from_hf("Plachta/Seed-VC",
                                                    "hift.pt",
                                                    "hifigan.yml")
    hift_config = yaml.safe_load(open(hift_config_path, 'r'))
    hift_gen = HiFTGenerator(**hift_config['hift'], f0_predictor=ConvRNNF0Predictor(**hift_config['f0_predictor']))
    hift_gen.load_state_dict(torch.load(hift_checkpoint_path, map_location='cpu'))
    hift_gen.eval()
    hift_gen.to(device)

    ckpt_path, config_path = load_custom_model_from_hf("Plachta/FAcodec", 'pytorch_model.bin', 'config.yml')

    codec_config = yaml.safe_load(open(config_path))
    codec_model_params = recursive_munch(codec_config['model_params'])
    codec_encoder = build_model(codec_model_params, stage="codec")

    ckpt_params = torch.load(ckpt_path, map_location="cpu")

    for key in codec_encoder:
        codec_encoder[key].load_state_dict(ckpt_params[key], strict=False)
    _ = [codec_encoder[key].eval() for key in codec_encoder]
    _ = [codec_encoder[key].to(device) for key in codec_encoder]

    # Generate mel spectrograms
    mel_fn_args = {
        "n_fft": config['preprocess_params']['spect_params']['n_fft'],
        "win_size": config['preprocess_params']['spect_params']['win_length'],
        "hop_size": config['preprocess_params']['spect_params']['hop_length'],
        "num_mels": config['preprocess_params']['spect_params']['n_mels'],
        "sampling_rate": sr,
        "fmin": 0,
        "fmax": 8000,
        "center": False
    }
    from modules.audio import mel_spectrogram

    to_mel = lambda x: mel_spectrogram(x, **mel_fn_args)
    max_context_window = sr // hop_length * 30
    overlap_frame_len = 64
    overlap_wave_len = overlap_frame_len * hop_length
    

    return  model, to_mel, hift_gen, campplus_model, codec_encoder, sr, max_context_window, overlap_wave_len

def crossfade(chunk1, chunk2, overlap):
    fade_out = np.linspace(1, 0, overlap)
    fade_in = np.linspace(0, 1, overlap)
    chunk2[:overlap] = chunk2[:overlap] * fade_in + chunk1[-overlap:] * fade_out
    return chunk2


@torch.no_grad()
@torch.inference_mode()
def voice_conversion(source, target, diffusion_steps, length_adjust, inference_cfg_rate, n_quantizers):
    model, to_mel, hift_gen, campplus_model, codec_encoder, sr, max_context_window, overlap_wave_len = load_models()
    inference_module = model 
    mel_fn = to_mel
    bitrate = "320k"
    overlap_frame_len = 64
    # Load audio
    source_audio = librosa.load(source, sr=sr)[0]
    ref_audio = librosa.load(target, sr=sr)[0]

    # Process audio
    source_audio = torch.tensor(source_audio).unsqueeze(0).float().to(device)
    ref_audio = torch.tensor(ref_audio[:sr * 25]).unsqueeze(0).float().to(device)

    # Resample
    ref_waves_16k = torchaudio.functional.resample(ref_audio, sr, 16000)

    # Extract features
    converted_waves_24k = torchaudio.functional.resample(source_audio, sr, 24000)
    waves_input = converted_waves_24k.unsqueeze(1)
    max_wave_len_per_chunk = 24000 * 20
    wave_input_chunks = [
        waves_input[..., i:i + max_wave_len_per_chunk] for i in range(0, waves_input.size(-1), max_wave_len_per_chunk)
    ]
    S_alt_chunks = []
    for i, chunk in enumerate(wave_input_chunks):
        z = codec_encoder.encoder(chunk)
        (
            quantized,
            codes
        ) = codec_encoder.quantizer(
            z,
            chunk,
        )
        S_alt = torch.cat([codes[1], codes[0]], dim=1)
        S_alt_chunks.append(S_alt)
    S_alt = torch.cat(S_alt_chunks, dim=-1)

    # S_ori should be extracted in the same way
    waves_24k = torchaudio.functional.resample(ref_audio, sr, 24000)
    waves_input = waves_24k.unsqueeze(1)
    z = codec_encoder.encoder(waves_input)
    (
        quantized,
        codes
    ) = codec_encoder.quantizer(
        z,
        waves_input,
    )
    S_ori = torch.cat([codes[1], codes[0]], dim=1)

    mel = mel_fn(source_audio.to(device).float())
    mel2 = mel_fn(ref_audio.to(device).float())

    target_lengths = torch.LongTensor([int(mel.size(2) * length_adjust)]).to(mel.device)
    target2_lengths = torch.LongTensor([mel2.size(2)]).to(mel2.device)

    feat2 = torchaudio.compliance.kaldi.fbank(ref_waves_16k,
                                              num_mel_bins=80,
                                              dither=0,
                                              sample_frequency=16000)
    feat2 = feat2 - feat2.mean(dim=0, keepdim=True)
    style2 = campplus_model(feat2.unsqueeze(0))

    
    F0_ori = None
    F0_alt = None
    shifted_f0_alt = None

    # Length regulation
    cond = inference_module.length_regulator(S_alt, ylens=target_lengths, n_quantizers=int(n_quantizers), f0=shifted_f0_alt)[0]
    prompt_condition = inference_module.length_regulator(S_ori, ylens=target2_lengths, n_quantizers=int(n_quantizers), f0=F0_ori)[0]

    max_source_window = max_context_window - mel2.size(2)
    # split source condition (cond) into chunks
    processed_frames = 0
    generated_wave_chunks = []
    # generate chunk by chunk and stream the output
    while processed_frames < cond.size(1):
        chunk_cond = cond[:, processed_frames:processed_frames + max_source_window]
        is_last_chunk = processed_frames + max_source_window >= cond.size(1)
        cat_condition = torch.cat([prompt_condition, chunk_cond], dim=1)
        # Voice Conversion
        vc_target = inference_module.cfm.inference(cat_condition,
                                                   torch.LongTensor([cat_condition.size(1)]).to(mel2.device),
                                                   mel2, style2, None, diffusion_steps,
                                                   inference_cfg_rate=inference_cfg_rate)
        vc_target = vc_target[:, :, mel2.size(-1):]
        
        vc_wave = hift_gen.inference(vc_target, f0=None)
        
        if processed_frames == 0:
            if is_last_chunk:
                output_wave = vc_wave[0].cpu().numpy()
                generated_wave_chunks.append(output_wave)
                output_wave = (output_wave * 32768.0).astype(np.int16)
                mp3_bytes = AudioSegment(
                    output_wave.tobytes(), frame_rate=sr,
                    sample_width=output_wave.dtype.itemsize, channels=1
                ).export(format="mp3", bitrate=bitrate).read()
                yield mp3_bytes, (sr, np.concatenate(generated_wave_chunks))
                break
            output_wave = vc_wave[0, :-overlap_wave_len].cpu().numpy()
            generated_wave_chunks.append(output_wave)
            previous_chunk = vc_wave[0, -overlap_wave_len:]
            processed_frames += vc_target.size(2) - overlap_frame_len
            output_wave = (output_wave * 32768.0).astype(np.int16)
            mp3_bytes = AudioSegment(
                output_wave.tobytes(), frame_rate=sr,
                sample_width=output_wave.dtype.itemsize, channels=1
            ).export(format="mp3", bitrate=bitrate).read()
            yield mp3_bytes, None
        elif is_last_chunk:
            output_wave = crossfade(previous_chunk.cpu().numpy(), vc_wave[0].cpu().numpy(), overlap_wave_len)
            generated_wave_chunks.append(output_wave)
            processed_frames += vc_target.size(2) - overlap_frame_len
            output_wave = (output_wave * 32768.0).astype(np.int16)
            mp3_bytes = AudioSegment(
                output_wave.tobytes(), frame_rate=sr,
                sample_width=output_wave.dtype.itemsize, channels=1
            ).export(format="mp3", bitrate=bitrate).read()
            yield mp3_bytes, (sr, np.concatenate(generated_wave_chunks))
            break
        else:
            output_wave = crossfade(previous_chunk.cpu().numpy(), vc_wave[0, :-overlap_wave_len].cpu().numpy(), overlap_wave_len)
            generated_wave_chunks.append(output_wave)
            previous_chunk = vc_wave[0, -overlap_wave_len:]
            processed_frames += vc_target.size(2) - overlap_frame_len
            output_wave = (output_wave * 32768.0).astype(np.int16)
            mp3_bytes = AudioSegment(
                output_wave.tobytes(), frame_rate=sr,
                sample_width=output_wave.dtype.itemsize, channels=1
            ).export(format="mp3", bitrate=bitrate).read()
            yield mp3_bytes, None
            
    
    del model, hift_gen, campplus_model, codec_encoder
    gc.collect()
    torch.cuda.empty_cache()
    