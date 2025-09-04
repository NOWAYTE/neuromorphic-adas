MODEL_PATHS = {
    'weights': 'models/final_model_weights.pth',
    'config': 'models/audio_config.json',
    'onnx': 'models/model.onnx',
    'thermal_weights': 'models/thermal_encoder.pth'
}

DATA_PATHS = {
    'neuromorphic': 'data/events/',
    'acoustic': 'data/audio/',
    'thermal': 'data/thermal/'
}

# Model parameters
MODEL_CONFIG = {
    'num_classes': 3,  # normal, siren, hazard
    'thermal': {
        'input_size': (224, 224),
        'feature_dim': 128,
        'normalize': True
    },
    'audio': {
        'sample_rate': 16000,
        'n_fft': 512,
        'n_mels': 64,
        'hop_length': 160
    },
    'event': {
        'time_window': 50.0,  # ms
        'height': 260,
        'width': 346
    }
}
