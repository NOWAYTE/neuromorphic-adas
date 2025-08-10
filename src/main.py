import torch
import json
from models.hybrid_fusion import HybridFusionModel
from utils.audio_processor import AudioProcessor
from config import MODEL_PATHS

class ADASSystem:
    def __init__(self, use_onnx=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_onnx = use_onnx
        
        if use_onnx:
            self.init_onnx_runtime()
        else:
            self.init_pytorch_model()
        
        self.init_audio_processor()
    
    def init_pytorch_model(self):
        """Load PyTorch model from weights"""
        self.model = HybridFusionModel().to(self.device)
        
        # Load weights
        state_dict = torch.load(MODEL_PATHS['weights'], map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
    
    def init_onnx_runtime(self):
        """Initialize ONNX runtime"""
        import onnxruntime
        self.session = onnxruntime.InferenceSession(
            MODEL_PATHS['onnx'],
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
    
    def init_audio_processor(self):
        """Initialize audio processor with saved config"""
        with open(MODEL_PATHS['config'], 'r') as f:
            config = json.load(f)
        self.audio_processor = AudioProcessor(**config)
    
    def process_frame(self, event_data, audio_data):
        """Process a single frame of data"""
        if self.use_onnx:
            return self.process_with_onnx(event_data, audio_data)
        else:
            return self.process_with_pytorch(event_data, audio_data)
    
    def process_with_pytorch(self, event_data, audio_data):
        # Preprocess audio
        audio_features = self.audio_processor.extract_features(audio_data)
        
        # Convert to tensors
        event_tensor = torch.tensor(event_data).float().to(self.device)
        audio_tensor = torch.tensor(audio_features).float().to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(event_tensor, audio_tensor)
        
        return outputs
    
    def process_with_onnx(self, event_data, audio_data):
        # Preprocess audio
        audio_features = self.audio_processor.extract_features(audio_data)
        
        # Prepare inputs
        ort_inputs = {
            'events': event_data.numpy(),
            'audio': audio_features.numpy()
        }
        
        # Run inference
        ort_outs = self.session.run(None, ort_inputs)
        return ort_outs

if __name__ == "__main__":
    system = ADASSystem()
    
    # Example usage with dummy data
    dummy_events = torch.randn(1, 10, 2, 260, 346)
    dummy_audio = torch.randn(16000)  # 1 second audio
    
    outputs = system.process_frame(dummy_events, dummy_audio)
    print("System outputs:", outputs)
