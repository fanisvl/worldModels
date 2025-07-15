import torch
from torch.utils.data import Dataset
import numpy as np
import os

class RolloutDataset(Dataset):
    """
    Contains observation (64x64x3) and action data
    """

    def __init__(self, data_dir, transform=None, max_files=None):
        self.data_dir = data_dir
        self.transform = transform

        # get paths to all .npz files and optionally limit number of files
        all_files = sorted(
            os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npz')
        )
        self.file_paths = all_files[:max_files] if max_files is not None else all_files

        # build index of (file_index, observation_index) for all frames in selected files
        self.observation_idx = []  # list of tuples (file_idx, obs_idx)
        for file_idx, file_path in enumerate(self.file_paths):
            with np.load(file_path) as data:
                num_frames = data['observations'].shape[0]
            self.observation_idx.extend([(file_idx, i) for i in range(num_frames)])

    def __len__(self):
        return len(self.observation_idx)
    
    def __getitem__(self, idx):
        file_idx, obs_idx = self.observation_idx[idx]
        file_path = self.file_paths[file_idx]

        with np.load(file_path) as data:
            observation = data['observations'][obs_idx]  # (64, 64, 3)
            action = data['actions'][obs_idx]            # (action_dim,)

        # convert to float32, scale to [0,1], permute to (C, H, W)
        observation = torch.from_numpy(observation).float() / 255.0
        observation = observation.permute(2, 0, 1)

        if self.transform:
            observation = self.transform(observation)
        
        return observation, action, file_idx
    

class LatentSequenceDataset(Dataset):
    def __init__(self, data_dir, sequence_length=500):
        self.data_dir = data_dir
        self.sequence_length = sequence_length

        # get paths to all .npz files
        self.file_paths = sorted([
            os.path.join(data_dir, f) for f in os.listdir(data_dir)
            if f.endswith('.npz')
        ])

        # (file_index, start_frame_index)
        self.indices = []
        for file_idx, file_path in enumerate(self.file_paths):
            with np.load(file_path) as data:
                num_frames = data['latent_observations'].shape[0]

                # seq_len inputs and seq_len outputs 
                # The targer for input t is the latent at t+1
                # A sequences of length L requires L+1 total frames
                # The last possible start index is num_frames - (L+1)
                if num_frames > self.sequence_length:
                    for i in range(num_frames - self.sequence_length):
                        self.indices.append((file_idx, i))

        
    def __len__(self):
        """ Total number of possible sequences """
        return len(self.indices)

    def __getitem__(self, idx):
        """
        Returns a single (input_sequence, target_sequence) pair
        """
        file_idx, start_idx = self.indices[idx]
        file_path = self.file_paths[file_idx]

        with np.load(file_path) as data:
            end_idx = start_idx + self.sequence_length

            # We need latents from t=0 to t=L and actions from t=0 to t=L-1
            latents = data['latent_observations'][start_idx:end_idx+1]
            actions = data['actions'][start_idx:end_idx]

            # Input (latent_t, action_t)
            # Target (latent_{t+1})
            x_latents = latents[:-1] # (L, latent_dim+action_dim)
            x = np.concatenate((x_latents, actions), axis=-1)

            # Target Sequence (L, latent_dim)
            y = latents[1:]

            x_tensor = torch.from_numpy(x).float()
            y_tensor = torch.from_numpy(y).float()
            
            return x_tensor, y_tensor


