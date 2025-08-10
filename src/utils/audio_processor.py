import librosa
import numpy as np

class AudioProcessor:
    def __init__(self, sample_rate=16000, n_fft=1024, hop_length=512, n_mels=64):
        self.sr = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        
    def extract_features(self, audio):
        """Extract mel spectrogram features from audio"""
        # Ensure correct sample rate
        if isinstance(audio, torch.Tensor):
            audio = audio.numpy()
        
        # Handle different audio lengths
        if len(audio) < self.sr:
            audio = np.pad(audio, (0, max(0, self.sr - len(audio))))
        
        # Extract features
        S = librosa.feature.melspectrogram(
            y=audio, 
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )
        return librosa.power_to_db(S, ref=np.max)
