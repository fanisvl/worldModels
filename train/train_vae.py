import sys
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from datetime import datetime
from tqdm import tqdm
import os
import subprocess
import wandb

sys.path.append("worldModels")
from data.dataset import RolloutDataset
from modules.vae import VAE

# -- Settings --
DATA_DIR = '/content/vizdoom_10k'
VAL_DIR = '/content/vizdoom_val_1k'
BATCH_SIZE = 256
NUM_WORKERS = 2
MAX_SAMPLES = None
EPOCHS = 1
LATENT_DIM = 32
LR = 1e-3

ddmm = datetime.now().strftime("%d-%m")
RUN_NAME = f'vae.lat{LATENT_DIM}.e{EPOCHS}.sample{MAX_SAMPLES}.{ddmm}'

# -- Datasets & Loaders --
train_dataset = RolloutDataset(data_dir=DATA_DIR, max_samples=MAX_SAMPLES)
val_dataset = RolloutDataset(data_dir=VAL_DIR)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -- W&B Init --
wandb.init(
    project="VAE",
    name=RUN_NAME,
    config={
        "latent_dim": LATENT_DIM,
        "learning_rate": LR,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "MAX_SAMPLES": MAX_SAMPLES
    }
)

# -- Model & Optimizer --
model = VAE(latent_dim=LATENT_DIM).to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)

# Utility: GPU stats
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

# -- Training & Validation Loops --
for epoch in range(1, EPOCHS + 1):
    model.train()
    train_total, train_recon, train_kl = 0, 0, 0
    global_step = (epoch - 1) * len(train_loader)

    for obs, action, idx in tqdm(train_loader, desc=f"Train Epoch {epoch}/{EPOCHS}"):
        x = obs.to(device)
        recon_x, mu, log_var, _ = model(x)
        recon_loss = torch.nn.functional.mse_loss(recon_x, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        loss = recon_loss + kl_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_total += loss.item()
        train_recon += recon_loss.item()
        train_kl += kl_loss.item()

        gpu_u, mem_u = get_gpu_stats()
        log = {
            "train/total_loss": loss.item(),
            "train/recon_loss": recon_loss.item(),
            "train/kl_loss": kl_loss.item(),
            "step": global_step
        }
        if gpu_u is not None:
            log.update({"gpu/util": gpu_u, "gpu/mem": mem_u})
        wandb.log(log)
        global_step += 1

    # Compute training averages
    n_train = len(train_loader.dataset)
    avg_train = train_total / n_train
    avg_recon = train_recon / n_train
    avg_kl = train_kl / n_train

    # -- Validation --
    model.eval()
    val_total, val_recon, val_kl = 0, 0, 0
    with torch.no_grad():
        for obs, action, idx in tqdm(val_loader, desc=f"Val Epoch {epoch}/{EPOCHS}"):
            x = obs.to(device)
            recon_x, mu, log_var, _ = model(x)
            recon_l = torch.nn.functional.mse_loss(recon_x, x, reduction='sum')
            kl_l = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            val_total += (recon_l + kl_l).item()
            val_recon += recon_l.item()
            val_kl += kl_l.item()

    # Compute validation averages
    n_val = len(val_loader.dataset)
    avg_val = val_total / n_val
    avg_val_recon = val_recon / n_val
    avg_val_kl = val_kl / n_val

    # Log epoch metrics
    print(f"Epoch {epoch}: train_loss={avg_train:.4f}, val_loss={avg_val:.4f}")
    wandb.log({
        "epoch": epoch,
        "epoch/train_loss": avg_train,
        "epoch/val_loss": avg_val,
        "epoch/train_recon": avg_recon,
        "epoch/val_recon": avg_val_recon,
        "epoch/train_kl": avg_kl,
        "epoch/val_kl": avg_val_kl
    })

    # Save checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    cp = f"{RUN_NAME}.pt"
    torch.save(model.state_dict(), cp)
    art = wandb.Artifact('vae-checkpoints', type='model')
    art.add_file(cp)
    wandb.log_artifact(art)

wandb.finish()
