import torch
import torch.nn.functional as F
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        
        # -- ENCODER --
        # Input:  N, 3, 64, 64
        # Output: N, 2*latent_dim (mu, log_sigma)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2), # -> N, 32, 31, 31
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), # -> N, 64, 14, 14
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2), # -> N,128,6,6
            nn.ReLU(),
            nn.Conv2d(128, 256,4, stride=2), # -> N,256,2,2
            nn.ReLU(),
            nn.Flatten(start_dim=1), # N, 256*2*2 = 1024
            nn.Linear(256*2*2, 2*self.latent_dim), # N,1024 -> N, 2*latent_dim
        )

        # -- DECODER --
        # Input: N, latent_dim
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256*2*2), # (N, 256*2*2)
            nn.Unflatten(1, (1024, 1, 1)), # (N, 1024, 1, 1)
            nn.ConvTranspose2d(1024, 128, 5, stride=2), # -> (N, 128, 5, 5)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 5, stride=2), # -> N, 64, 13, 13
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 6, stride=2), # -> N, 32, 30, 30
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 6, stride=2), # -> N, 3, 64, 64
            nn.Sigmoid()
        )

    def forward(self, x):
        # encoder
        mu, log_var = torch.split(self.encoder(x), self.latent_dim, dim=1) # (N, latent_dim), (N, latent_dim)

        # reparameterization trick
        eps = torch.randn_like(log_var)
        z = mu + torch.exp(0.5 * log_var) * eps # -> latent_dim

        # decoder
        recon_x =  self.decoder(z)

        return recon_x, mu, log_var
    
def vae_loss(recon_x, x, mu, log_var):
    reconstruction_loss = F.mse_loss(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - torch.exp(log_var)) 
    loss = reconstruction_loss + kl_loss
    return loss