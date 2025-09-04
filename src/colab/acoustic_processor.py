import librosa
import numpy as np
import torch

class AudioProcessor:
    def __init__(self, sample_rate=16000, n_fft=1024, hop_length=512, n_mels=64):
        self.sr = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        
    def load_audio(self, file_path):
        y, _ = librosa.load(file_path, sr=self.sr)
        return y
        
    def extract_features(self, audio):
        """Extract mel spectrogram features"""
        # Handle short audio clips
        if len(audio) < self.n_fft:
            audio = np.pad(audio, (0, self.n_fft - len(audio)))
            
        S = librosa.feature.melspectrogram(
            y=audio, 
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )
        log_S = librosa.power_to_db(S, ref=np.max)
        return torch.tensor(log_S, dtype=torch.float32)
    
    def augment_audio(self, audio, noise_level=0.005):
        """Add realistic noise augmentation"""
        noise = np.random.normal(0, noise_level, len(audio))
        return audio + noise