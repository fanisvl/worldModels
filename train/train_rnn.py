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
from modules.mdn_rnn import MDN_RNN, rnn_combined_loss
from data.dataset import LatentSequenceDataset

# -- Argument Parser --
parser = argparse.ArgumentParser(description="MDN-RNN Training")
parser.add_argument('--data_dir', type=str, required=True, help='Directory for training data (latents)')
parser.add_argument('--load_pretrained', type=str, default=None, help='Start from a pre-trained model')
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
parser.add_argument('--latent_done_ratio', default=None, type=float, help='Ratio of latent loss to done loss for balancing.')
args = parser.parse_args()


DATA_DIR = args.data_dir
LOAD_PRETRAINED = args.load_pretrained
DESC = args.desc
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
LATENT_DONE_RATIO = args.latent_done_ratio

print("=" * 50)
print("TRAINING CONFIGURATION")
print("=" * 50)
print(f"Data directory: {DATA_DIR}")
print(f"Load pretrained: {LOAD_PRETRAINED}")
print(f"Description: {DESC}")
print(f"Number of workers: {NUM_WORKERS}")
print(f"Checkpoint interval: {CHECKPOINT_INTERVAL}")
print(f"Seed: {SEED}")
print(f"Epochs: {EPOCHS}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Sequence length: {SEQUENCE_LENGTH}")
print(f"Validation split: {VAL_SPLIT}")
print(f"Latent dimension: {LATENT_DIM}")
print(f"Action dimension: {ACTION_DIM}")
print(f"Hidden size: {HIDDEN_SIZE}")
print(f"Number of layers: {N_LAYERS}")
print(f"Number of Gaussians: {N_GAUSSIANS}")
print(f"Learning rate: {LR}")
print(f"Latent/Done Loss Ratio: {LATENT_DONE_RATIO}")
print("=" * 50)

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
BCE_POS_WEIGHT = dataset.pos_weight.to(device)

# -- W&B Init --
os.environ["WANDB_LOG_GPU_PERFORMANCE"] = "true"
wandb.login()
wandb.init(
    project="mdn-rnn",
    name=RUN_NAME,
    notes=DESC,
    config={
        "data_dir": DATA_DIR,
        "load_pretrained": LOAD_PRETRAINED,
        "num_workers": NUM_WORKERS,
        "checkpoint_interval": CHECKPOINT_INTERVAL,
        "seed": SEED,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "sequence_length": SEQUENCE_LENGTH,
        "val_split": VAL_SPLIT,
        "latent_dim": LATENT_DIM,
        "action_dim": ACTION_DIM,
        "hidden_size": HIDDEN_SIZE,
        "n_layers": N_LAYERS,
        "n_gaussians": N_GAUSSIANS,
        "learning_rate": LR,
        "latent_done_ratio": LATENT_DONE_RATIO,
    }
)
config = wandb.config

# -- Model --
model = MDN_RNN(LATENT_DIM, ACTION_DIM, HIDDEN_SIZE, N_LAYERS, N_GAUSSIANS).to(device)

if os.path.exists(LOAD_PRETRAINED):
    print(f"Loading pre-trained weights from {LOAD_PRETRAINED}...")
    model.load_state_dict(torch.load(LOAD_PRETRAINED, map_location=device), strict=False)
    print("Weights loaded successfully.")

opt = torch.optim.Adam(model.parameters(), LR)
scheduler = ReduceLROnPlateau(opt, 'min', factor=0.5, patience=5)
print(f'Total params: {sum(p.numel() for p in model.parameters())}')

# -- Training Loop --
global_step = 0

def train():
    global global_step
    model.train()
    total_loss = 0.0
    total_latent_loss = 0.0
    total_terminal_loss = 0.0
    total_terminal_loss_unweighted = 0.0
    total_done_loss_weight = 0.0

    for x, y in tqdm(train_loader, desc="Training"):
        start_time = time.time()
        x = x.to(device, non_blocking=True)
        y = y = {k: v.to(device, non_blocking=True) for k, v in y.items()}
        data_time_ms = int((time.time() - start_time) * 1000)
        start_time = time.time()
        opt.zero_grad()
        pi_logits, mu, sigma_logits, done_logits, _ = model(x)
        combined_loss, latent_loss, done_loss_weight, terminal_loss_unweighted, terminal_loss = rnn_combined_loss(pi_logits, mu, sigma_logits, done_logits, y, BCE_POS_WEIGHT, LATENT_DONE_RATIO)
        combined_loss.backward()
        model_time_ms = int((time.time() - start_time) * 1000)

        # Clip gradients and log pre-clip norm and clip coefficient
        total_norm_unscaled = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        clip_coef = 1.0 / (total_norm_unscaled + 1e-6) if total_norm_unscaled > 1.0 else 1.0

        wandb.log({
            "batch/loss": combined_loss.item(),
            "batch/latent_loss": latent_loss.item(),
            "batch/terminal_loss_weighted": terminal_loss.item(),
            "batch/terminal_loss_unweighted": terminal_loss_unweighted.item(),
            "batch/done_loss_weight": done_loss_weight,
            "batch/grad_norm_unclipped": total_norm_unscaled,
            "batch/grad_clip_coef": clip_coef,
            "profile/data_time_ms": data_time_ms,
            "profile/model_time_ms": model_time_ms,
        }, step=global_step)

        opt.step()
        total_loss += combined_loss.item()
        total_latent_loss += latent_loss.item()
        total_terminal_loss += terminal_loss.item()
        total_terminal_loss_unweighted += terminal_loss_unweighted.item()
        total_done_loss_weight += done_loss_weight
        global_step += 1
    
    num_batches = len(train_loader)
    return (total_loss / num_batches, 
            total_latent_loss / num_batches, 
            total_terminal_loss / num_batches,
            total_terminal_loss_unweighted / num_batches,
            total_done_loss_weight / num_batches)

@torch.no_grad()
def validate():
    model.eval()
    combined_loss = 0.0
    latent_loss = 0.0
    terminal_loss = 0.0
    terminal_loss_unweighted = 0.0
    
    for x, y in tqdm(val_loader, desc="Validating"):
        x = x.to(device)
        y = y = {k: v.to(device, non_blocking=True) for k, v in y.items()}
        pi_logits, mu, sigma_logits, done_logits, _ = model(x)
        c_loss, l_loss, _, t_loss_unweighted, t_loss = rnn_combined_loss(pi_logits, mu, sigma_logits, done_logits, y, BCE_POS_WEIGHT, LATENT_DONE_RATIO)
        combined_loss += c_loss.item()
        latent_loss += l_loss.item()
        terminal_loss += t_loss.item()
        terminal_loss_unweighted += t_loss_unweighted.item()

    num_batches = len(val_loader)
    val_loss = combined_loss / num_batches
    avg_latent_loss = latent_loss / num_batches
    avg_terminal_loss = terminal_loss / num_batches
    avg_terminal_loss_unweighted = terminal_loss_unweighted / num_batches
    
    return val_loss, avg_latent_loss, avg_terminal_loss, avg_terminal_loss_unweighted

# -- Run Training --
for epoch in range(EPOCHS):
    train_loss, train_latent_loss, train_terminal_loss, train_terminal_loss_unweighted, avg_done_weight = train()
    val_loss, val_latent_loss, val_terminal_loss, val_terminal_loss_unweighted = validate()

    # Step the scheduler
    scheduler.step(val_loss)

    # Log epoch-level metrics
    wandb.log({
        "epoch": epoch + 1,
        "epoch/train_loss": train_loss,
        "epoch/train_latent_loss": train_latent_loss,
        "epoch/train_terminal_loss_weighted": train_terminal_loss,
        "epoch/train_terminal_loss_unweighted": train_terminal_loss_unweighted,
        "epoch/avg_done_loss_weight": avg_done_weight,
        "epoch/val_loss": val_loss,
        "epoch/val_latent_loss": val_latent_loss,
        "epoch/val_terminal_loss_weighted": val_terminal_loss,
        "epoch/val_terminal_loss_unweighted": val_terminal_loss_unweighted,
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