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
from modules.mdn_rnn import MDN_RNN, rnn_loss
from data.dataset import LatentSequenceDataset
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from modules.vae import VAE
import torch.nn.functional as F

# -- Argument Parser --
parser = argparse.ArgumentParser(description="MDN-RNN Training")
parser.add_argument('--data_dir', type=str, required=True, help='Directory for training data (latents)')
parser.add_argument('--num_workers', type=int, default=2, help='Number of workers for data loading')
parser.add_argument('--checkpoint_interval', type=int, default=None, help='Save a model checkpoint every N epochs')
parser.add_argument('--seed', type=int, default=99, help='Random seed for reproducibility')
parser.add_argument('--val_split', type=float, default=0.2, help='Validation split percentage')

# Model Hyperparams
parser.add_argument('--epochs', type=int, required=True, help='Number of epochs to train for')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
parser.add_argument('--sequence_length', type=int, default=500, help='Sequence length for training')
parser.add_argument('--latent_dim', type=int, default=32, help='Dimensionality of the latent space')
parser.add_argument('--action_dim', type=int, default=3, help='Dimensionality of the action space')
parser.add_argument('--hidden_size', type=int, default=256, help='Size of the RNN hidden state')
parser.add_argument('--n_layers', type=int, default=1, help='Number of layers in the RNN')
parser.add_argument('--n_gaussians', type=int, default=5, help='Number of Gaussians in the mixture density network')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')

# dream visualization args
parser.add_argument('--val_data_dir', type=str, help='Directory for validation data (raw rollouts)')
parser.add_argument('--vae_path', type=str, help='Path to the trained VAE model for decoding dreams')
parser.add_argument('--dream_context_frames', type=int, default=3, help='Number of context frames for the dream')
parser.add_argument('--dream_length', type=int, default=100, help='Total length of the dream sequence to generate')

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
scheduler = ReduceLROnPlateau(opt, 'min', factor=0.5, patience=5, verbose=True)
print(f'Total params: {sum(p.numel() for p in model.parameters())}')

# == DREAM VISUALIZATION LOGIC ==
vae = VAE(latent_dim=LATENT_DIM, inverted=True).to(device)
vae.load_state_dict(torch.load(args.vae_path, map_location=device))
vae.eval()
print("VAE for decoding dreams loaded successfully.")

# We also need a fixed validation sequence to dream from every time
val_rollout_path = os.path.join(args.val_data_dir, sorted(os.listdir(args.val_data_dir))[0])
val_rollout = np.load(val_rollout_path)
val_obs = torch.from_numpy(val_rollout['observations'][:args.dream_length]).float().to(device)
val_actions = torch.from_numpy(val_rollout['actions'][:args.dream_length]).float().to(device)
print(f"Loaded fixed validation sequence for dreaming from {val_rollout_path}")

def _sample_from_mdn(pi_logits, mu, sigma_logits, temperature=1.0):
    """Samples a latent vector from the MDN output with temperature."""
    if temperature <= 0:
        print(f'[WARN] Temperature was <= 0, setting it to 1e-8')
        temperature = 1e-8

    pi = torch.softmax(pi_logits / temperature, dim=-1)
    sigma = torch.exp(sigma_logits)
    mixture = torch.distributions.Categorical(probs=pi)
    k = mixture.sample()

    # Sample from chosen Gaussian
    mu_k = mu[k]
    sigma_k = sigma[k]
    z_next = torch.normal(mu_k, sigma_k)

    return z_next.unsqueeze(0)

@torch.no_grad()
def generate_dream_sequence(rnn_model, context_frames, actions_for_dream, temperature=0.1):
    num_context = len(context_frames)
    num_dream_frames = len(actions_for_dream) - num_context + 1
    dreamed_latents = []

    # 1. Encode context frames
    context_z = vae.encode(context_frames.permute(0, 3, 1, 2) / 255.0)

    # 2. Prime RNN with context
    # Note: Assumes action_dim is 3 for one-hot encoding
    context_actions_onehot = F.one_hot(actions_for_dream[:num_context-1].long(), num_classes=args.action_dim).float()
    rnn_input_context = torch.cat([context_z[:-1], context_actions_onehot], dim=-1).unsqueeze(0)
    _, _, _, _, (h, c) = rnn_model(rnn_input_context)

    current_z = context_z[-1].unsqueeze(0)

    # 3. Generate dream frames
    for i in range(num_dream_frames):
        action_idx = num_context - 1 + i
        action_onehot = F.one_hot(actions_for_dream[action_idx].long(), num_classes=args.action_dim).float().unsqueeze(0)
        
        rnn_input = torch.cat([current_z, action_onehot], dim=-1).unsqueeze(0)
        pi_logits, mu, log_sigma, _, (h, c) = rnn_model(rnn_input, (h, c))
        
        z_next = _sample_from_mdn(pi_logits, mu, log_sigma, temperature)
        dreamed_latents.append(z_next)
        current_z = z_next

    return torch.cat(dreamed_latents, dim=0)


def create_comparison_animation(ground_truth, rnn_model, save_path):
    # This function now takes the rnn_model directly
    context_frames_obs = ground_truth[:args.dream_context_frames]
    
    # Generate VAE reconstruction and RNN dream
    decoded_gt = vae.decode(vae.encode(ground_truth.permute(0, 3, 1, 2) / 255.0)).permute(0, 2, 3, 1).cpu().numpy()
    dreamed_latents = generate_dream_sequence(rnn_model, context_frames_obs, val_actions)
    dreamed_frames = vae.decode(dreamed_latents).permute(0, 2, 3, 1).cpu().numpy()
    
    ground_truth_np = ground_truth.cpu().numpy()
    actions_np = val_actions.cpu().numpy()

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    def update_frame(frame):
        fig.suptitle(f'Epoch {epoch+1} - Frame {frame}', fontsize=16)
        ax1.clear(); ax1.imshow(ground_truth_np[frame]); ax1.set_title(f'Ground Truth'); ax1.axis('off')
        ax2.clear(); ax2.imshow(decoded_gt[frame]); ax2.set_title(f'VAE Reconstruction'); ax2.axis('off')
        ax3.clear()
        if frame < args.dream_context_frames:
            ax3.imshow(decoded_gt[frame])
            ax3.set_title(f'RNN Context', color='orange')
        else:
            dream_idx = frame - args.dream_context_frames
            if dream_idx < len(dreamed_frames):
                ax3.imshow(dreamed_frames[dream_idx])
            ax3.set_title(f'RNN Dream')
        ax3.axis('off')

    anim = FuncAnimation(fig, update_frame, frames=len(ground_truth_np), interval=100)
    anim.save(save_path, writer=PillowWriter(fps=10))
    plt.close(fig)
    return save_path

# =============================================================================

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
        pi_logits, mu, sigma_logits, done_logits, _ = model(x)
        latent_loss, terminal_loss, combined_loss = rnn_loss(pi_logits, mu, sigma_logits, done_logits, y)
        combined_loss.backward()
        model_time_ms = int((time.time() - start_time) * 1000)

        # Clip gradients and log pre-clip norm and clip coefficient
        total_norm_unscaled = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        clip_coef = 1.0 / (total_norm_unscaled + 1e-6) if total_norm_unscaled > 1.0 else 1.0

        wandb.log({
            "batch/loss": combined_loss.item(),
            'batch/latent_loss': latent_loss.item(),
            'batch/terminal_loss': terminal_loss.item(),
            "batch/grad_norm_unclipped": total_norm_unscaled,
            "batch/grad_clip_coef": clip_coef,
            "profile/data_time_ms": data_time_ms,
            "profile/model_time_ms": model_time_ms,
        }, step=global_step)

        opt.step()
        total_loss += combined_loss.item()
        global_step += 1
    return total_loss / len(train_loader)

@torch.no_grad()
def validate():
    model.eval()
    total_val_loss = 0.0
    total_latent_loss = 0.0
    total_terminal_loss = 0.0
    
    for x, y in tqdm(val_loader, desc="Validating"):
        x, y = x.to(device), y.to(device)
        pi_logits, mu, sigma_logits, done_logits, _ = model(x)
        
        latent_loss, terminal_loss, combined_loss = rnn_loss(pi_logits, mu, sigma_logits, done_logits, y)

        total_val_loss += combined_loss.item()
        total_latent_loss += latent_loss.item()
        total_terminal_loss += terminal_loss.item()

    # Calculate averages
    avg_val_loss = total_val_loss / len(val_loader)
    avg_latent_loss = total_latent_loss / len(val_loader)
    avg_terminal_loss = total_terminal_loss / len(val_loader)

    wandb.log({
        "epoch/val_loss_combined": avg_val_loss,
        "epoch/val_loss_latent": avg_latent_loss,
        "epoch/val_loss_terminal": avg_terminal_loss,
    }, step=global_step)

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

        # Generate dream to evaluate RNN
        print(f"\nGenerating dream for epoch {epoch+1}...")
        model.eval()
        dream_path = os.path.join("dreams", f"{RUN_NAME}_epoch_{epoch+1}.gif")
        os.makedirs("dreams", exist_ok=True)
        create_comparison_animation(val_obs, model, save_path=dream_path)
        wandb.log({
            "epoch/dream_visualization": wandb.Video(dream_path, fps=10, format="gif")
        }, step=global_step)

# -- Save Final Model --
model_path = RUN_NAME + ".pt"
torch.save(model.state_dict(), model_path)

# -- Log Model to W&B --
artifact = wandb.Artifact(RUN_NAME, type="model")
artifact.add_file(model_path)
wandb.log_artifact(artifact)
wandb.finish()