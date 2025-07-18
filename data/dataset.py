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
    """
    Dataset class that pre-loads all data from .npz files into RAM
    to accelerate training by removing I/O bottlenecks.
    """
    def __init__(self, data_dir, sequence_length=100):
        self.sequence_length = sequence_length

        # Pre-load all data into memory
        file_paths = sorted([
            os.path.join(data_dir, f) for f in os.listdir(data_dir)
            if f.endswith('.npz')
        ])

        # This list will hold all episode data (latents and actions) as tensors in RAM
        self.episodes = []
        print("Pre-loading all data into memory, this might take a moment...")
        for file_path in tqdm(file_paths, desc='Pre-loading data into RAM'):
            with np.load(file_path) as data:
                # Convert to tensors immediately
                latents = torch.from_numpy(data['latent_observations']).float()
                actions = torch.from_numpy(data['actions']).float()
                self.episodes.append({'latents': latents, 'actions': actions})
        print("Data pre-loading complete.")

        # Create sequence indices from the in-memory data
        self.indices = []
        dropped_files = 0
        total_files = len(self.episodes)
        
        print("Generating sequence indices...")
        for episode_idx, episode_data in enumerate(tqdm(self.episodes, desc='Generating indices')):
            num_frames = episode_data['latents'].shape[0]

            if num_frames > self.sequence_length:
                # Each start index yields one full sequence of length `sequence_length`
                for i in range(num_frames - self.sequence_length):
                    # Store the episode index and the start frame index
                    self.indices.append((episode_idx, i))
            else:
                dropped_files += 1

        print(f'[LatentSequenceDataset] '
              f'Loaded {len(self.indices)} sequences from {total_files - dropped_files} files. '
              f'Discarded {dropped_files}/{total_files} files smaller than {self.sequence_length} frames.')

    def __len__(self):
        """ Total number of possible sequences. """
        return len(self.indices)

    def __getitem__(self, idx):
        """
        Returns a single (input_sequence, target_sequence) pair.
        """
        # Retrieve the pre-calculated episode and start frame index
        episode_idx, start_idx = self.indices[idx]
        
        # Get the corresponding episode data (already in memory as tensors)
        episode = self.episodes[episode_idx]
        
        end_idx = start_idx + self.sequence_length

        # Slice the tensors to get the required window.
        # We need latents from t=0 to t=L and actions from t=0 to t=L-1.
        latents = episode['latents'][start_idx : end_idx + 1]
        actions = episode['actions'][start_idx : end_idx].view(-1, 1)

        # Create input `x` and target `y`
        # Input: (latent_t, action_t) for t in [0, L-1]
        # Target: latent_{t+1} for t in [0, L-1]
        x_latents = latents[:-1]
        x = torch.cat((x_latents, actions), dim=-1)
        y = latents[1:]
            
        return x, y


