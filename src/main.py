import torch
import json
from models.hybrid_fusion import HybridFusionModel
from utils.audio_processor import AudioProcessor
from colab.thermal_processor import ThermalProcessor
from config import MODEL_PATHS, DATA_PATHS

class ADASSystem:
    def __init__(self, use_onnx=False, use_thermal=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_onnx = use_onnx
        self.use_thermal = use_thermal
        
        if use_onnx:
            self.init_onnx_runtime()
        else:
            self.init_pytorch_model()
        
        self.init_audio_processor()
        if self.use_thermal:
            self.init_thermal_processor()
    
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
    
    def init_thermal_processor(self):
        """Initialize thermal image processor"""
        self.thermal_processor = ThermalProcessor()
    
    def preprocess_thermal(self, thermal_image):
        """Preprocess thermal image data"""
        if not self.use_thermal:
            return None
        return self.thermal_processor.preprocess(thermal_image).to(self.device)
    
    def process_frame(self, event_data, audio_data, thermal_data=None):
        """Process a single frame of data"""
        if self.use_thermal and thermal_data is None:
            raise ValueError("Thermal data is required when use_thermal=True")
            
        if self.use_onnx:
            return self.process_with_onnx(event_data, audio_data, thermal_data)
        else:
            return self.process_with_pytorch(event_data, audio_data, thermal_data)
    
    def process_with_pytorch(self, event_data, audio_data, thermal_data=None):
        # Preprocess audio
        audio_input = self.audio_processor.process(audio_data).to(self.device)
        
        # Preprocess thermal if available
        thermal_input = None
        if self.use_thermal and thermal_data is not None:
            thermal_input = self.preprocess_thermal(thermal_data)
        
        with torch.no_grad():
            # Add batch dimension if needed
            if len(event_data.shape) == 4:  # [C, H, W] -> [1, C, H, W]
                event_data = event_data.unsqueeze(0)
            if len(audio_input.shape) == 1:  # [T] -> [1, T]
                audio_input = audio_input.unsqueeze(0)
            
            # Move to device
            event_data = event_data.to(self.device)
            audio_input = audio_input.to(self.device)
            
            # Forward pass
            classification, confidence = self.model(
                event_data, 
                audio_input,
                thermal_input
            )
            
            # Get predictions
            probs = torch.softmax(classification, dim=1)
            pred_class = torch.argmax(probs, dim=1)
            
            return {
                'class': pred_class.item(),
                'confidence': confidence.item(),
                'probabilities': probs.cpu().numpy()[0]
            }
    
    def process_with_onnx(self, event_data, audio_data, thermal_data=None):
        # Convert inputs to numpy arrays for ONNX
        event_np = event_data.cpu().numpy() if torch.is_tensor(event_data) else event_data
        audio_np = audio_data.cpu().numpy() if torch.is_tensor(audio_data) else audio_data
        
        # Prepare input dictionary
        ort_inputs = {
            self.session.get_inputs()[0].name: event_np,
            self.session.get_inputs()[1].name: audio_np
        }
        
        # Add thermal input if available
        if self.use_thermal and thermal_data is not None:
            thermal_np = thermal_data.cpu().numpy() if torch.is_tensor(thermal_data) else thermal_data
            ort_inputs[self.session.get_inputs()[2].name] = thermal_np
        
        # Run inference
        ort_outs = self.session.run(None, ort_inputs)
        
        # Process outputs
        classification = torch.from_numpy(ort_outs[0])
        confidence = torch.from_numpy(ort_outs[1])
        
        # Get predictions
        probs = torch.softmax(classification, dim=1)
        pred_class = torch.argmax(probs, dim=1)
        
        return {
            'class': pred_class.item(),
            'confidence': confidence.item(),
            'probabilities': probs.numpy()[0]
        }

if __name__ == "__main__":
    system = ADASSystem()
    
    # Example usage with dummy data
    dummy_events = torch.randn(1, 10, 2, 260, 346)
    dummy_audio = torch.randn(16000)  # 1 second audio
    dummy_thermal = torch.randn(1, 3, 224, 224)  # dummy thermal image
    
    outputs = system.process_frame(dummy_events, dummy_audio, dummy_thermal)
    print("System outputs:", outputs)
