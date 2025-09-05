import torch
import numpy as np
import time
from typing import Dict, List, Optional
import logging

class ADASInferencePipeline:
    def __init__(self, model_path: str, config: Dict, use_onnx: bool = False):
        """
        Complete ADAS inference pipeline for production deployment
        
        Args:
            model_path: Path to trained model weights
            config: Configuration dictionary
            use_onnx: Whether to use ONNX runtime for inference
        """
        self.config = config
        self.use_onnx = use_onnx
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize components
        self._load_model(model_path)
        self._initialize_processors()
        self._initialize_safety_controller()
        
        # Performance monitoring
        self.inference_times = []
        self.anomaly_scores = []
        
        logging.info(f"ADAS Pipeline initialized on {self.device}")
    
    def _load_model(self, model_path: str):
        """Load the trained model"""
        if self.use_onnx:
            import onnxruntime as ort
            self.session = ort.InferenceSession(
                model_path,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            logging.info("ONNX model loaded successfully")
        else:
            self.model = torch.jit.load(model_path) if model_path.endswith('.pt') else self._load_pytorch_model(model_path)
            self.model.to(self.device)
            self.model.eval()
            logging.info("PyTorch model loaded successfully")
    
    def _load_pytorch_model(self, model_path: str):
        """Load PyTorch model with architecture definition"""
        # Model architecture should match your training
        model = HybridFusionModel(
            num_classes=self.config['model']['num_classes'],
            thermal_feat_dim=self.config['model']['thermal_feat_dim']
        )
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        return model
    
    def _initialize_processors(self):
        """Initialize all data processors"""
        self.audio_processor = AudioProcessor(**self.config['audio'])
        self.event_processor = EventDataLoader(**self.config['events'])
        self.thermal_processor = ThermalProcessor(**self.config['thermal'])
        
    def _initialize_safety_controller(self):
        """Initialize safety control system"""
        self.safety_controller = SafetyController(**self.config['safety'])
        
    def process_frame(self, sensor_data: Dict) -> Dict:
        """
        Process a single frame of multi-sensor data
        
        Args:
            sensor_data: Dictionary containing:
                - events: Neuromorphic event data
                - audio: Raw audio data
                - thermal: Thermal image data
                - timestamp: Frame timestamp
                
        Returns:
            Dictionary with inference results and safety decisions
        """
        start_time = time.time()
        
        try:
            # Preprocess each modality
            processed_data = self._preprocess_data(sensor_data)
            
            # Run inference
            inference_result = self._run_inference(processed_data)
            
            # Apply safety rules and anomaly detection
            safety_decision = self._apply_safety_rules(inference_result, processed_data)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            self.inference_times.append(processing_time)
            
            # Prepare output
            output = {
                **inference_result,
                **safety_decision,
                'processing_time': processing_time,
                'timestamp': sensor_data.get('timestamp', time.time())
            }
            
            # Log if anomaly detected
            if safety_decision['anomaly_detected']:
                self._log_anomaly(output)
                
            return output
            
        except Exception as e:
            logging.error(f"Error processing frame: {e}")
            return self._handle_error_state()
    
    def _preprocess_data(self, sensor_data: Dict) -> Dict:
        """Preprocess all sensor data"""
        processed = {}
        
        # Process event data
        if 'events' in sensor_data:
            processed['events'] = self.event_processor.events_to_tensor(sensor_data['events'])
            processed['events'] = processed['events'].to(self.device)
        
        # Process audio data
        if 'audio' in sensor_data:
            processed['audio'] = self.audio_processor.extract_features(sensor_data['audio'])
            processed['audio'] = processed['audio'].to(self.device)
        
        # Process thermal data
        if 'thermal' in sensor_data:
            processed['thermal'] = self.thermal_processor.preprocess(sensor_data['thermal'])
            processed['thermal'] = processed['thermal'].to(self.device)
            
        return processed
    
    def _run_inference(self, processed_data: Dict) -> Dict:
        """Run model inference on processed data"""
        if self.use_onnx:
            return self._run_onnx_inference(processed_data)
        else:
            return self._run_pytorch_inference(processed_data)
    
    def _run_pytorch_inference(self, processed_data: Dict) -> Dict:
        """Run inference using PyTorch model"""
        with torch.no_grad():
            # Prepare inputs
            events = processed_data.get('events', torch.zeros(1, 10, 2, 260, 346).to(self.device))
            audio = processed_data.get('audio', torch.zeros(1, 64, 64).to(self.device))
            thermal = processed_data.get('thermal', torch.zeros(1, 1, 224, 224).to(self.device))
            
            # Run model
            classification, confidence = self.model(events, audio, thermal)
            
            # Process outputs
            probs = torch.softmax(classification, dim=1)
            pred_class = torch.argmax(probs, dim=1)
            
            return {
                'prediction': pred_class.item(),
                'confidence': confidence.item(),
                'probabilities': probs.cpu().numpy()[0].tolist(),
                'raw_output': classification.cpu().numpy()
            }
    
    def _run_onnx_inference(self, processed_data: Dict) -> Dict:
        """Run inference using ONNX runtime"""
        # Prepare inputs for ONNX
        ort_inputs = {}
        
        if 'events' in processed_data:
            ort_inputs['events'] = processed_data['events'].cpu().numpy()
        if 'audio' in processed_data:
            ort_inputs['audio'] = processed_data['audio'].cpu().numpy()
        if 'thermal' in processed_data:
            ort_inputs['thermal'] = processed_data['thermal'].cpu().numpy()
        
        # Run inference
        ort_outs = self.session.run(None, ort_inputs)
        
        # Process outputs
        classification = torch.from_numpy(ort_outs[0])
        confidence = torch.from_numpy(ort_outs[1])
        
        probs = torch.softmax(classification, dim=1)
        pred_class = torch.argmax(probs, dim=1)
        
        return {
            'prediction': pred_class.item(),
            'confidence': confidence.item(),
            'probabilities': probs.numpy()[0].tolist(),
            'raw_output': classification.numpy()
        }
    
    def _apply_safety_rules(self, inference_result: Dict, processed_data: Dict) -> Dict:
        """Apply safety rules and anomaly detection"""
        # Calculate anomaly score
        anomaly_score = self._calculate_anomaly_score(inference_result, processed_data)
        self.anomaly_scores.append(anomaly_score)
        
        # Get safety decision
        safety_decision = self.safety_controller.evaluate(
            anomaly_score=anomaly_score,
            model_confidence=inference_result['confidence'],
            prediction=inference_result['prediction']
        )
        
        return {
            'anomaly_score': anomaly_score,
            'anomaly_detected': anomaly_score > self.config['safety']['anomaly_threshold'],
            'safety_actions': safety_decision
        }
    
    def _calculate_anomaly_score(self, inference_result: Dict, processed_data: Dict) -> float:
        """Calculate comprehensive anomaly score across modalities"""
        # 1. Model confidence-based anomaly
        confidence_anomaly = 1.0 - inference_result['confidence']
        
        # 2. Modality-specific anomaly detection
        modality_anomalies = []
        
        if 'events' in processed_data:
            event_anomaly = self._detect_event_anomalies(processed_data['events'])
            modality_anomalies.append(event_anomaly)
            
        if 'audio' in processed_data:
            audio_anomaly = self._detect_audio_anomalies(processed_data['audio'])
            modality_anomalies.append(audio_anomaly)
            
        if 'thermal' in processed_data:
            thermal_anomaly = self._detect_thermal_anomalies(processed_data['thermal'])
            modality_anomalies.append(thermal_anomaly)
        
        # 3. Combined anomaly score (weighted average)
        modality_anomaly = np.mean(modality_anomalies) if modality_anomalies else 0
        
        # Final anomaly score (prioritize model confidence)
        anomaly_score = 0.7 * confidence_anomaly + 0.3 * modality_anomaly
        
        return anomaly_score
    
    def _detect_event_anomalies(self, event_data: torch.Tensor) -> float:
        """Detect anomalies in event data"""
        # Calculate event density
        event_density = torch.mean(event_data)
        
        # Check for abnormal event patterns
        temporal_consistency = self._check_temporal_consistency(event_data)
        
        # Combine metrics
        return float(1.0 - (0.5 * event_density + 0.5 * temporal_consistency))
    
    def _detect_audio_anomalies(self, audio_data: torch.Tensor) -> float:
        """Detect anomalies in audio data"""
        # Calculate audio energy
        energy = torch.mean(audio_data ** 2)
        
        # Check for abnormal frequency patterns
        spectral_flatness = self._calculate_spectral_flatness(audio_data)
        
        # Combine metrics
        return float(0.3 * (1.0 - energy) + 0.7 * spectral_flatness)
    
    def _detect_thermal_anomalies(self, thermal_data: torch.Tensor) -> float:
        """Detect anomalies in thermal data"""
        # Calculate thermal gradients
        gradients = torch.abs(torch.gradient(thermal_data.view(thermal_data.shape[2:]))[0])
        gradient_score = torch.mean(gradients)
        
        # Check for abnormal temperature patterns
        temp_std = torch.std(thermal_data)
        
        # Combine metrics
        return float(0.6 * gradient_score + 0.4 * temp_std)
    
    def _log_anomaly(self, output: Dict):
        """Log anomaly details for analysis"""
        logging.warning(
            f"Anomaly detected: "
            f"score={output['anomaly_score']:.3f}, "
            f"prediction={output['prediction']}, "
            f"confidence={output['confidence']:.3f}"
        )
        
        # Save detailed anomaly data if needed
        if self.config.get('save_anomalies', False):
            self._save_anomaly_data(output)
    
    def _handle_error_state(self) -> Dict:
        """Handle errors gracefully and return safe state"""
        return {
            'prediction': 0,  # Default to normal state
            'confidence': 0.0,
            'anomaly_score': 1.0,  # High anomaly score to trigger safety
            'anomaly_detected': True,
            'safety_actions': self.safety_controller.get_emergency_actions(),
            'processing_time': 0.0,
            'timestamp': time.time(),
            'error': True
        }
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        if not self.inference_times:
            return {}
            
        times = np.array(self.inference_times)
        return {
            'avg_inference_time': np.mean(times),
            'max_inference_time': np.max(times),
            'min_inference_time': np.min(times),
            'num_processed_frames': len(self.inference_times),
            'anomaly_rate': np.mean(np.array(self.anomaly_scores) > self.config['safety']['anomaly_threshold'])
        }
    
    def shutdown(self):
        """Cleanup resources"""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'session'):
            del self.session
        torch.cuda.empty_cache()