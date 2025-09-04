import h5py
import numpy as np
import torch

class EventDataLoader:
    def __init__(self, time_window=50.0, height=260, width=346):
        """
        Neuromorphic event data loader
        :param time_window: Time window in milliseconds
        """
        self.time_window = time_window * 1000  # Convert to µs
        self.height = height
        self.width = width
        
    def load_events(self, file_path):
        with h5py.File(file_path, 'r') as f:
            return {
                't': np.array(f['events/t']),
                'x': np.array(f['events/x']),
                'y': np.array(f['events/y']),
                'p': np.array(f['events/p'])
            }
            
    def events_to_tensor(self, events):
        """Convert events to tensor representation"""
        min_t = np.min(events['t'])
        max_t = np.max(events['t'])
        num_frames = int(np.ceil((max_t - min_t) / self.time_window))
        
        # Initialize tensor: [frames, channels, height, width]
        tensor = torch.zeros((num_frames, 2, self.height, self.width))
        
        for frame_idx in range(num_frames):
            start_t = min_t + frame_idx * self.time_window
            end_t = start_t + self.time_window
            
            # Find events in current time window
            mask = (events['t'] >= start_t) & (events['t'] < end_t)
            frame_events = {k: v[mask] for k, v in events.items()}
            
            # Accumulate events
            for t, x, y, p in zip(frame_events['t'], frame_events['x'], 
                                  frame_events['y'], frame_events['p']):
                channel = 0 if p > 0 else 1
                if 0 <= y < self.height and 0 <= x < self.width:
                    tensor[frame_idx, channel, y, x] += 1
                    
        return tensor

    def normalize_events(self, tensor):
        """Normalize event counts per frame"""
        # Add small epsilon to avoid division by zero
        frame_sums = tensor.sum(dim=(1, 2, 3), keepdim=True) + 1e-8
        return tensor / frame_sums
        