import torch
import torchaudio
from modules.commons import build_model, load_checkpoint, recursive_munch
import yaml
from hf_utils import load_custom_model_from_hf
import numpy as np
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


