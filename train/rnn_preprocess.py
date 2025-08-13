import sys
import torch
import numpy as np
import os
import argparse
from glob import glob
from tqdm import tqdm
sys.path.append("worldModels")
sys.path.append(".")
from modules.vae import VAE


def _load_rollout(file_path):
    data = np.load(file_path)
    frames = data['observations']
    actions = data['actions']
    terminals = data['terminals']
    return frames, actions, terminals


def _prepare_frames(frames):
    """
    Convert frames (uint8 HWC or NCHW) to float32 normalized tensor NCHW.
    """
    arr = frames
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32) / 255.0
    # If shape is (N, H, W, C) -> transpose to (N, C, H, W)
    if arr.ndim == 4 and arr.shape[-1] in (1, 3):
        arr = np.transpose(arr, (0, 3, 1, 2))
    return arr


def encode_rollout(vae, device, frames_np, batch_size, use_fp16=False):
    frames_np = _prepare_frames(frames_np)
    latents = []
    with torch.no_grad():
        for i in range(0, len(frames_np), batch_size):
            batch = torch.from_numpy(frames_np[i:i+batch_size]).to(device)
            if use_fp16:
                batch = batch.half()
            _, _, _, z = vae(batch)
            latents.append(z.float().cpu().numpy())
    return np.concatenate(latents, axis=0)

def precompute_latents_streaming(vae_path, data_dir, output_dir, batch_size, fp16=False, overwrite=False):
    """
    Memory-efficient: iterate rollout files one-by-one, encode, and save.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    vae = VAE(32)
    vae.load_state_dict(torch.load(vae_path, map_location=device))
    if fp16:
        vae.half()
    vae.to(device)
    vae.eval()

    os.makedirs(output_dir, exist_ok=True)

    files = sorted(glob(os.path.join(data_dir, "rollout_*.npz")))
    if not files:
        raise RuntimeError(f"No rollout_*.npz files found in {data_dir}")

    for f in tqdm(files, desc="Encoding rollouts"):
        out_f = os.path.join(output_dir, os.path.basename(f))
        if (not overwrite) and os.path.exists(out_f):
            continue
        frames, actions, terminals = _load_rollout(f)
        latents = encode_rollout(vae, device, frames, batch_size, use_fp16=fp16)
        np.savez_compressed(out_f, latent_observations=latents, actions=actions, terminals=terminals)
    print("Done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Precompute latents for MDN-RNN training (streaming, low RAM).')
    parser.add_argument('--vae_path', type=str, required=True, help='Path to the VAE model file')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing rollout_*.npz')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save latent rollouts')
    parser.add_argument('--batch_size', type=int, default=64, help='Inference batch size')
    parser.add_argument('--fp16', action='store_true', help='Use half precision for VAE (saves VRAM)')
    parser.add_argument('--overwrite', action='store_true', help='Re-encode even if output file exists')
    args = parser.parse_args()

    precompute_latents_streaming(
        args.vae_path,
        args.data_dir,
        args.output_dir,
        args.batch_size,
        fp16=args.fp16,
        overwrite=args.overwrite
    )