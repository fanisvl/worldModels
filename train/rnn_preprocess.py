import sys
from torch.utils.data import DataLoader
import torch
import numpy as np
import os
import argparse
from tqdm import tqdm
sys.path.append("worldModels")
sys.path.append(".")
from data.dataset import RolloutDataset
from modules.vae import VAE


def precompute_latents(vae_path, data_dir, output_dir, batch_size):
    """
    # Pre-process rollouts for MDN-RNN training by using the VAE to create the latent dataset.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # load the model
    vae = VAE(32)
    vae.load_state_dict(torch.load(vae_path, map_location=device))
    vae.to(device)
    vae.eval()

    # load dataset
    data = RolloutDataset(data_dir)
    dl = DataLoader(data, batch_size=batch_size)
    # Process each batch and accumulate latents by sequence
    sequence_data = {}  # Dictionary to store data by file_idx (sequence)
    
    for batch_frames, batch_actions, batch_file_idxs in tqdm(dl, desc='Encoding Observations'):
        batch_frames = batch_frames.to(device)
        # Forward pass through VAE
        with torch.no_grad():
            _, _, _, z = vae(batch_frames)  # z: (batch_size, 32)
        
        z = z.cpu()
        # Group by sequence (file_idx)
        for i in range(len(batch_file_idxs)):
            file_idx = batch_file_idxs[i].item()
            
            if file_idx not in sequence_data:
                sequence_data[file_idx] = {
                    'latents': [],
                    'actions': []
                }
            
            sequence_data[file_idx]['latents'].append(z[i].numpy())
            sequence_data[file_idx]['actions'].append(batch_actions[i].numpy())
    
    # Save each sequence as a separate file
    os.makedirs(output_dir, exist_ok=True)
    for seq_idx, seq_data in sequence_data.items():
        data = {
            'latent_observations': np.array(seq_data['latents']),
            'actions': np.array(seq_data['actions'])
        }
        
        np.savez_compressed(
            os.path.join(output_dir, f'rollout_{seq_idx:05d}.npz'),
            **data
        )


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Precompute latents for MDN-RNN training')
    parser.add_argument('--vae_path', type=str, help='Path to the VAE model file')
    parser.add_argument('--data_dir', type=str, help='Directory containing the input data')
    parser.add_argument('--output_dir', type=str, help='Directory to save the latent data')
    parser.add_argument('--batch_size', type=int, required=False, default=32, help='Inference batch size')
    args = parser.parse_args()
    precompute_latents(args.vae_path, args.data_dir, args.output_dir, args.batch_size)