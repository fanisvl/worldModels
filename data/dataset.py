import torch
from torch.utils.data import Dataset
import numpy as np
import os

class RolloutDataset(Dataset):
    """
    Contains observation (64x64x3) and action data
    """

    def __init__(self, data_dir, transform=None, max_samples=None):
        self.data_dir = data_dir
        self.transform = transform

        # get paths to all .npz files
        self.file_paths = sorted([
            os.path.join(data_dir, f) for f in os.listdir(data_dir)
            if f.endswith('.npz')
        ])

        # preload up to max_samples frame indices to avoid loading full dataset into memory
        self.observation_idx = []  # (file_index, observation_idx)
        total = 0
        for file_idx, file_path in enumerate(self.file_paths):
            with np.load(file_path) as data:
                n = data['observations'].shape[0]
            # how many from this file?
            take = n
            if max_samples is not None:
                remaining = max_samples - total
                take = min(n, remaining)
            # extend by only `take` entries
            self.observation_idx.extend([(file_idx, i) for i in range(take)])
            total += take
            # stop if we've reached the quota
            if max_samples is not None and total >= max_samples:
                break

    def __len__(self):
        return len(self.observation_idx)
    
    def __getitem__(self, idx):
        rollout_idx, observation_idx = self.observation_idx[idx]
        file_path = self.file_paths[rollout_idx]

        with np.load(file_path) as data:
            observation = data['observations'][observation_idx]  # (64, 64, 3)
            action = data['actions'][observation_idx]            # (action_dim,)

        # convert to float32, scale to [0,1], permute to (C, H, W)
        observation = torch.from_numpy(observation).float() / 255.0
        observation = observation.permute(2, 0, 1)

        if self.transform:
            observation = self.transform(observation)
        
        return (observation, action, rollout_idx)
    

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


