import torch
from torch.utils.data import Dataset
import numpy as np
import os

class CarRacingDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        # get paths to all .npz files
        self.file_paths = sorted([
            os.path.join(data_dir, f) for f in os.listdir(data_dir)
            if f.endswith('.npz')
        ])

        # preload frame indices to avoid loading full dataset into memory
        self.frame_index = [] # (file_index, frame_index)
        for file_idx, file_path in enumerate(self.file_paths):
            with np.load(file_path) as data:
                num_frames = data['observations'].shape[0]
                self.frame_index.extend([(file_idx, i) for i in range(num_frames)])

    def __len__(self):
        return len(self.frame_index)
    
    def __getitem__(self, idx):
        file_idx, frame_idx = self.frame_index[idx]
        file_path = self.file_paths[file_idx]

        with np.load(file_path) as data:
            frame = data['observations'][frame_idx] # (64, 64, 3)

        # convert to float32, scale to [0,1], permute to (3, 64, 64)
        frame = torch.from_numpy(frame).float() / 255.0
        frame = frame.permute(2,0,1)

        if self.transform:
            frame = self.transform(frame)
        
        return frame
