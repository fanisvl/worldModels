import sys
import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datetime import datetime
from tqdm import tqdm
import os
import wandb
import argparse
import random
import numpy as np
import time
sys.path.append("worldModels")
sys.path.append(".")
from modules.mdn_rnn import MDN_RNN, mdn_loss
from data.dataset import LatentSequenceDataset

# -- Argument Parser --
parser = argparse.ArgumentParser(description="MDN-RNN Training")
parser.add_argument('--data_dir', type=str, required=True, help='Directory for training data (latents)')
parser.add_argument("--desc", type=str, default="", help="Short description of the experiment")
parser.add_argument('--num_workers', type=int, default=2, help='Number of workers for data loading')
parser.add_argument('--checkpoint_interval', type=int, default=None, help='Save a model checkpoint every N epochs')
parser.add_argument('--seed', type=int, default=99, help='Random seed for reproducibility')
parser.add_argument('--epochs', type=int, required=True, help='Number of epochs to train for')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
parser.add_argument('--sequence_length', type=int, default=500, help='Sequence length for training')
parser.add_argument('--val_split', type=float, default=0.2, help='Validation split percentage')
parser.add_argument('--latent_dim', type=int, default=32, help='Dimensionality of the latent space')
parser.add_argument('--action_dim', type=int, default=3, help='Dimensionality of the action space')
parser.add_argument('--hidden_size', type=int, default=256, help='Size of the RNN hidden state')
parser.add_argument('--n_layers', type=int, default=1, help='Number of layers in the RNN')
parser.add_argument('--n_gaussians', type=int, default=5, help='Number of Gaussians in the mixture density network')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
args = parser.parse_args()

DATA_DIR = args.data_dir
NUM_WORKERS = args.num_workers
CHECKPOINT_INTERVAL = args.checkpoint_interval
SEED = args.seed
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
SEQUENCE_LENGTH = args.sequence_length
VAL_SPLIT = args.val_split
LATENT_DIM = args.latent_dim
ACTION_DIM = args.action_dim
HIDDEN_SIZE = args.hidden_size
N_LAYERS = args.n_layers
N_GAUSSIANS = args.n_gaussians
LR = args.lr
ddmm = datetime.now().strftime("%d-%m")
dataset_name = DATA_DIR.split('/')[-1]
RUN_NAME = f'rnn.lat{LATENT_DIM}.nl.{N_LAYERS}.h{HIDDEN_SIZE}.seq{SEQUENCE_LENGTH}.e{EPOCHS}.bs{BATCH_SIZE}.{dataset_name}.{ddmm}'

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(SEED)

# -- Datasets & Loaders --
dataset = LatentSequenceDataset(DATA_DIR, sequence_length=SEQUENCE_LENGTH)
val_size = int(len(dataset) * VAL_SPLIT)
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(SEED)
)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -- W&B Init --
os.environ["WANDB_LOG_GPU_PERFORMANCE"] = "true"
wandb.login()
wandb.init(
    project="mdn-rnn",
    name=RUN_NAME,
    notes=args.desc,
    config={
        "latent_dim": LATENT_DIM,
        "action_dim": ACTION_DIM,
        "hidden_size": HIDDEN_SIZE,
        "n_layers": N_LAYERS,
        "n_gaussians": N_GAUSSIANS,
        "learning_rate": LR,
        "batch_size": BATCH_SIZE,
        "sequence_length": SEQUENCE_LENGTH,
        "epochs": EPOCHS,
        "seed": SEED,
    }
)
config = wandb.config

# -- Model --
model = MDN_RNN(LATENT_DIM, ACTION_DIM, HIDDEN_SIZE, N_LAYERS, N_GAUSSIANS).to(device)
opt = torch.optim.Adam(model.parameters(), LR)
scheduler = ReduceLROnPlateau(opt, 'min', factor=0.5, patience=5)
print(f'Total params: {sum(p.numel() for p in model.parameters())}')

# -- Training Loop --
global_step = 0

def train():
    global global_step
    model.train()
    total_loss = 0.0

    for x, y in tqdm(train_loader, desc="Training"):
        start_time = time.time()
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        data_time_ms = int((time.time() - start_time) * 1000)
        start_time = time.time()
        opt.zero_grad()
        pi, mu, sigma, _ = model(x)
        loss = mdn_loss(pi, mu, sigma, y)
        loss.backward()
        model_time_ms = int((time.time() - start_time) * 1000)

        # Clip gradients and log pre-clip norm and clip coefficient
        total_norm_unscaled = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        clip_coef = 1.0 / (total_norm_unscaled + 1e-6) if total_norm_unscaled > 1.0 else 1.0

        wandb.log({
            "batch/loss": loss.item(),
            "batch/grad_norm_unclipped": total_norm_unscaled,
            "batch/grad_clip_coef": clip_coef,
            "profile/data_time_ms": data_time_ms,
            "profile/model_time_ms": model_time_ms,
        }, step=global_step)

        opt.step()
        total_loss += loss.item()
        global_step += 1
    return total_loss / len(train_loader)

@torch.no_grad()
def validate():
    model.eval()
    val_loss = 0.0
    for x, y in tqdm(val_loader, desc="Validating"):
        x, y = x.to(device), y.to(device)
        pi, mu, sigma, hidden = model(x)
        val_loss += mdn_loss(pi, mu, sigma, y).item()
    avg_val_loss = val_loss / len(val_loader)
    return avg_val_loss

# -- Run Training --
for epoch in range(EPOCHS):
    train_loss = train()
    val_loss = validate()

    # Step the scheduler
    scheduler.step(val_loss)

    # Log epoch-level metrics
    wandb.log({
        "epoch": epoch + 1,
        "epoch/train_loss": train_loss,
        "epoch/val_loss": val_loss,
        "epoch/lr": opt.param_groups[0]['lr']
    }, step=global_step)

    print(f'Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}')

    # -- Checkpointing --
    if CHECKPOINT_INTERVAL and (epoch + 1) % CHECKPOINT_INTERVAL == 0:
        os.makedirs('checkpoints', exist_ok=True)
        checkpoint_path = f"checkpoints/{RUN_NAME}.cpt.{epoch+1}.pt"
        torch.save(model.state_dict(), checkpoint_path)
        wandb.save(checkpoint_path)

# -- Save Final Model --
model_path = RUN_NAME + ".pt"
torch.save(model.state_dict(), model_path)

# -- Log Model to W&B --
artifact = wandb.Artifact(RUN_NAME, type="model")
artifact.add_file(model_path)
wandb.log_artifact(artifact)
wandb.finish()