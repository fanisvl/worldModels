import sys
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from datetime import datetime
from tqdm import tqdm
import os
import sys
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Subset
import subprocess
import wandb

ddmm = datetime.now().strftime("%d-%m")

sys.path.append("worldModels")
from data.dataset import RolloutDataset
from modules.vae import VAE

# -- Dataset --
DATA_DIR = '/content/vizdoom_10k'
BATCH_SIZE = 256
NUM_WORKERS = 2

dataset = RolloutDataset(data_dir=DATA_DIR, max_files=500)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -- Hyperparams --
LATENT_DIM = 32
LR = 1e-3
EPOCHS = 1
RUN_NAME = f'vae.lat{LATENT_DIM}.e{EPOCHS}.sample500.{ddmm}'

# -- W&B Init --
wandb.init(
    project="VAE",
    name = RUN_NAME,
    config={
        "latent_dim": LATENT_DIM,
        "learning_rate": LR,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
    }
)

model = VAE(latent_dim=LATENT_DIM).to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)
model.train()

global_step = 0

# Utility: get GPU stats via nvidia-smi

def get_gpu_stats():
    try:
        output = subprocess.check_output([
            "nvidia-smi",
            "--query-gpu=utilization.gpu,utilization.memory",
            "--format=csv,nounits,noheader"
        ])
        gpu_str, mem_str = output.decode().strip().split(',')
        return int(gpu_str), int(mem_str)
    except Exception:
        return None, None

for epoch in range(EPOCHS):
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0

    for observation, action, rollout_idx in tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        batch = observation.to(device)

        recon_x, mu, log_var, z = model(batch)

        # Loss
        recon_loss = torch.nn.functional.mse_loss(recon_x, batch, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        loss = recon_loss + kl_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_kl_loss += kl_loss.item()

        # GPU stats
        gpu_util, mem_util = get_gpu_stats()

        # W&B logging per step
        log_dict = {
            "Loss/Total": loss.item(),
            "Loss/Reconstruction": recon_loss.item(),
            "Loss/KL": kl_loss.item(),
            "step": global_step,
        }
        if gpu_util is not None:
            log_dict.update({
                "GPU/Utilization(%)": gpu_util,
                "GPU/Memory_Util(%)": mem_util
            })

        wandb.log(log_dict)
        global_step += 1

    avg_loss = total_loss / len(dataloader.dataset)
    avg_recon = total_recon_loss / len(dataloader.dataset)
    avg_kl = total_kl_loss / len(dataloader.dataset)

    tqdm.write(f"Epoch {epoch+1}: Total Loss={avg_loss:.2f}, Recon Loss={avg_recon:.2f}, KL Loss={avg_kl:.2f}")

    # W&B logging per epoch
    wandb.log({
        "Epoch/Loss/Total": avg_loss,
        "Epoch/Loss/Reconstruction": avg_recon,
        "Epoch/Loss/KL": avg_kl,
        "epoch": epoch + 1
    })

    # Save checkpoint & upload to W&B
    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_path = f"checkpoints/vae_e{epoch+1}.pt"
    torch.save(model.state_dict(), checkpoint_path)
    # Upload as W&B Artifact
    artifact = wandb.Artifact('vae-checkpoints', type='model')
    artifact.add_file(checkpoint_path)
    wandb.log_artifact(artifact)

wandb.finish()
