import torch
from torch.utils.data import Dataset
import numpy as np
import os
import random
from tqdm import tqdm

class RolloutDataset(Dataset):
    """
    Contains observation (64x64x3) and action data
    """

    def __init__(self, data_dir, transform=None, max_samples=None, invert_colors=False):
        self.data_dir = data_dir
        self.transform = transform
        self.invert_colors = invert_colors

        # get paths to all .npz files
        self.file_paths = sorted([
            os.path.join(data_dir, f) for f in os.listdir(data_dir)
            if f.endswith('.npz')
        ])

        all_indices = []
        for file_idx, file_path in enumerate(tqdm(self.file_paths, desc='Loading RolloutDataset')):
            with np.load(file_path) as data:
                n = data['observations'].shape[0]
            all_indices.extend([(file_idx, i) for i in range(n)])

        if max_samples is not None and max_samples < len(all_indices):
            self.observation_idx = random.sample(all_indices, max_samples)
        else:
            self.observation_idx = all_indices
        
        print(f'[Rollout Dataset]\nLoaded {len(self.observation_idx)}/{len(all_indices)} samples.\nTotal Files/Rollouts: {len(self.file_paths)}')

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

        if self.invert_colors:
            observation = 1.0 - observation

        if self.transform:
            observation = self.transform(observation)
        
        return (observation, action, rollout_idx)
    

class LatentSequenceDataset(Dataset):
    def __init__(self, data_dir, sequence_length=500):
        self.data_dir = data_dir
        self.sequence_length = sequence_length

        # get all npz files
        self.file_paths = sorted([
            os.path.join(data_dir, f) for f in os.listdir(data_dir)
            if f.endswith('.npz')
        ])

        # collect sequence‐start indices, count dropped files
        self.indices = []
        dropped_files = 0
        for file_idx, file_path in enumerate(tqdm(self.file_paths, desc='Loading LatentSequence dataset')):
            with np.load(file_path) as data:
                num_frames = data['latent_observations'].shape[0]

                if num_frames > self.sequence_length:
                    # each start index yields one full L‐length sequence
                    for i in range(num_frames - self.sequence_length):
                        self.indices.append((file_idx, i))
                else:
                    dropped_files += 1

        total_files = len(self.file_paths)
        print(f'[LatentSequenceDataset] '
              f'Loaded {len(self.indices)} sequences from {total_files - dropped_files} files. '
              f'Discarded {dropped_files}/{total_files} files smaller than {self.sequence_length} frames.')

        
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
            actions = data['actions'][start_idx:end_idx].reshape(-1,1)

            # Input (latent_t, action_t)
            # Target (latent_{t+1})
            x_latents = latents[:-1] # (L, latent_dim+action_dim)
            x = np.concatenate((x_latents, actions), axis=-1)

            # Target Sequence (L, latent_dim)
            y = latents[1:]

            x_tensor = torch.from_numpy(x).float()
            y_tensor = torch.from_numpy(y).float()
            
            return x_tensor, y_tensor


