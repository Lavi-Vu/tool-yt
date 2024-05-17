from PIL import Image
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, QTimer
from PyQt5.uic import loadUi
from PyQt5.QtWidgets import QMainWindow,QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QTextEdit, QFileDialog, QComboBox, QMessageBox
import sys
from torch import no_grad, LongTensor
import torch
import numpy as np
import os
from text import text_to_sequence, _clean_text
import commons
from mel_processing import spectrogram_torch
import utils
from models import SynthesizerTrn

class TTSapp(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        loadUi(r"/home/lavi/Documents/myprj/text-to-speech-TTS-desktop-app/src/ui/SecondWindow.ui", self)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.modelPath = 'OUTPUT_MODEL/G_latest.pth'
        self.configPath = 'OUTPUT_MODEL/config.json'
    
    
    def initModel(self):
        self.hps = utils.get_hparams_from_file(self.configPath)
        self.net_g = SynthesizerTrn(
        len(self.hps.symbols),
        self.hps.data.filter_length // 2 + 1,
        self.hps.train.segment_size // self.hps.data.hop_length,
        n_speakers=self.hps.data.n_speakers,
        **self.hps.model).to(self.device)
        _ = self.net_g.eval()

        _ = utils.load_checkpoint(self.modelPath, self.net_g, None)
        self.speaker_ids = self.hps.speakers
        self.speakers = list(self.hps.speakers.keys())
        
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = TTSapp()
    window.show()
    sys.exit(app.exec_())