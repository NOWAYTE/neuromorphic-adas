%%writefile utils/neuromorphic_loader.py
import h5py
import numpy as np
import torch
import os

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
        """
        Load events from file, supporting .h5, .dat, .aedat
        """
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        try:
            if ext in ['.h5', '.hdf5']:
                return self._load_events_h5(file_path)
            elif ext in ['.dat', '.txt', '.csv']:
                return self._load_events_dat(file_path)
            elif ext in ['.aedat']:
                return self._load_events_aedat(file_path)
            else:
                return self._load_events_autodetect(file_path)
        except Exception as e:
            print(f"Error loading events from {file_path}: {e}")
            return self._create_dummy_events()

    def _load_events_h5(self, file_path):
        """Load events from HDF5 file"""
        with h5py.File(file_path, 'r') as f:
            return {
                't': np.array(f['events/t']),
                'x': np.array(f['events/x']),
                'y': np.array(f['events/y']),
                'p': np.array(f['events/p'])
            }

    def _load_events_dat(self, file_path):
        """Load N-CARS binary .dat files"""
        # First try binary
        try:
            data = np.fromfile(file_path, dtype=np.int32)
            if len(data) % 4 != 0:
                raise ValueError("Binary .dat file malformed")
            events = {
                't': data[0::4],
                'x': data[1::4],
                'y': data[2::4],
                'p': data[3::4]
            }
            return events
        except:
            # Fallback to text format
            with open(file_path, 'r') as f:
                lines = f.readlines()
            data_start = 0
            for i, line in enumerate(lines):
                if not line.startswith('%'):
                    data_start = i
                    break
            t, x, y, p = [], [], [], []
            for line in lines[data_start:]:
                parts = line.strip().split()
                if len(parts) >= 4:
                    t.append(float(parts[0]))
                    x.append(int(parts[1]))
                    y.append(int(parts[2]))
                    p.append(int(parts[3]))
            return {
                't': np.array(t),
                'x': np.array(x),
                'y': np.array(y),
                'p': np.array(p)
            }

    def _load_events_aedat(self, file_path):
        """Load events from AEDAT file (simplified)"""
        with open(file_path, 'rb') as f:
            data = np.fromfile(f, dtype=np.uint32)
        t = data[0::4]
        x = data[1::4]
        y = data[2::4]
        p = data[3::4]
        return {'t': t, 'x': x, 'y': y, 'p': p}

    def _load_events_autodetect(self, file_path):
        """Try H5 -> DAT -> AEDAT"""
        for loader in [self._load_events_h5, self._load_events_dat, self._load_events_aedat]:
            try:
                return loader(file_path)
            except:
                continue
        print(f"Could not determine format of {file_path}, returning dummy events")
        return self._create_dummy_events()

    def _create_dummy_events(self):
        """Create dummy events for debugging"""
        num_events = 1000
        return {
            't': np.linspace(0, 1000000, num_events),
            'x': np.random.randint(0, self.width, num_events),
            'y': np.random.randint(0, self.height, num_events),
            'p': np.random.randint(0, 2, num_events)
        }

    def events_to_tensor(self, events):
        """Convert events to tensor representation"""
        min_t = np.min(events['t'])
        max_t = np.max(events['t'])
        if min_t == max_t:
            num_frames = 1
        else:
            num_frames = max(1, int(np.ceil((max_t - min_t) / self.time_window)))

        tensor = torch.zeros((num_frames, 2, self.height, self.width))

        for frame_idx in range(num_frames):
            start_t = min_t + frame_idx * self.time_window
            end_t = start_t + self.time_window
            mask = (events['t'] >= start_t) & (events['t'] < end_t)
            frame_events = {k: v[mask] for k, v in events.items()}

            for t, x, y, p in zip(frame_events['t'], frame_events['x'], frame_events['y'], frame_events['p']):
                channel = 0 if p > 0 else 1
                if 0 <= y < self.height and 0 <= x < self.width:
                    tensor[frame_idx, channel, y, x] += 1

        return tensor

    def normalize_events(self, tensor):
        """Normalize event counts per frame"""
        frame_sums = tensor.sum(dim=(1, 2, 3), keepdim=True) + 1e-8
        return tensor / frame_sums
