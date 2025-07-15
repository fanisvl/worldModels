import sys
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from datetime import datetime
from tqdm import tqdm
import os
import subprocess
import wandb
import argparse

sys.path.append("worldModels")
from data.dataset import RolloutDataset
from modules.vae import VAE

# -- Argument Parser --
parser = argparse.ArgumentParser(description="VAE Training")
# 
parser.add_argument('--data_dir', type=str, required=True, help='Directory for training data')
parser.add_argument('--val_dir', type=str, required=True, help='Directory for validation data')
parser.add_argument('--num_workers', type=int, default=2, help='Number of workers for data loading')
parser.add_argument('--checkpoint_interval', type=int, default=None, help='Save a model checkpoint every N epochs')

# Hyperparams 
parser.add_argument('--epochs', type=int, required=True, help='Number of epochs to train for')
parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
parser.add_argument('--max_samples', type=int, default=None, help='Maximum number of samples to use from the training dataset')
parser.add_argument('--latent_dim', type=int, default=32, help='Dimensionality of the latent space')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--beta', type=float, default=1.0, help='Final weight of the KL term (beta in beta-VAE)')
parser.add_argument('--kl_anneal_epochs', type=int, default=0, help='Number of epochs to anneal KL-divergence weight')

args = parser.parse_args()


# -- Settings --
DATA_DIR = args.data_dir
VAL_DIR = args.val_dir
BATCH_SIZE = args.batch_size
NUM_WORKERS = args.num_workers
MAX_SAMPLES = args.max_samples
EPOCHS = args.epochs
LATENT_DIM = args.latent_dim
LR = args.lr
CHECKPOINT_INTERVAL = args.checkpoint_interval
BETA = args.beta
KL_ANNEAL_EPOCHS = args.kl_anneal_epochs

ddmm = datetime.now().strftime("%d-%m")
RUN_NAME = f'vae.lat{LATENT_DIM}.e{EPOCHS}.bs{BATCH_SIZE}.sample{MAX_SAMPLES}.{ddmm}'

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
        "MAX_SAMPLES": MAX_SAMPLES,
        "checkpoint_interval": CHECKPOINT_INTERVAL,
        "beta": BETA,
        "kl_anneal_epochs": KL_ANNEAL_EPOCHS,
    }
)

# -- Model & Optimizer --
model = VAE(latent_dim=LATENT_DIM).to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)

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

        # KL Annealing
        if KL_ANNEAL_EPOCHS > 0:
            anneal_steps = KL_ANNEAL_EPOCHS * len(train_loader)
            current_beta = BETA * min(1.0, global_step / anneal_steps)
        else:
            current_beta = BETA
        
        loss = recon_loss + current_beta * kl_loss

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
            "train/beta": current_beta,
            "train/lr": optimizer.param_groups[0]['lr'],
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

    # Log validation reconstructions from start, middle, and end of the dataset
    with torch.no_grad():
        n_val = len(val_dataset)
        indices_to_log = [0, n_val // 2, n_val - 1]
        
        # Get original images
        originals_obs = [val_dataset[i][0] for i in indices_to_log]
        originals = torch.stack(originals_obs).to(device)
        
        # Get reconstructions
        reconstructions, _, _, _ = model(originals)
        
        wandb.log({
            "val/originals": [wandb.Image(img.cpu()) for img in originals],
            "val/reconstructions": [wandb.Image(img.cpu()) for img in reconstructions],
            "epoch": epoch
        })

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
    if (CHECKPOINT_INTERVAL != None and epoch % CHECKPOINT_INTERVAL == 0) or epoch == EPOCHS:
        os.makedirs("checkpoints", exist_ok=True)
        cp_path = os.path.join("checkpoints", f"{RUN_NAME}_epoch{epoch}.pt")
        torch.save(model.state_dict(), cp_path)
        
        art = wandb.Artifact('vae-checkpoints', type='model')
        art.add_file(cp_path)
        wandb.log_artifact(art, aliases=[f"epoch_{epoch}"])
        print(f"Saved checkpoint at epoch {epoch} to {cp_path}")

wandb.finish()
