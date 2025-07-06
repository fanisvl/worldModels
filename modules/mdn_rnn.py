import torch
import torch.nn as nn
import torch.nn.functional as F

class MDN_RNN(nn.Module):
    def __init__(self, latent_dim, action_dim, hidden_size, num_layers, n_gaussians):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.n_gaussians = n_gaussians

        # Input: (N, L, latent_dim + action_dim)
        self.lstm = nn.LSTM(latent_dim + action_dim, 
                            hidden_size, 
                            num_layers, 
                            batch_first=True)
        
        self.pi = nn.Linear(hidden_size, n_gaussians)
        self.mu = nn.Linear(hidden_size, n_gaussians * latent_dim)
        self.sigma = nn.Linear(hidden_size, n_gaussians * latent_dim)

    def forward(self, x, hidden=None):
        lstm_out, hidden = self.lstm(x, hidden)      # (N, L, H_out), hidden state
        logits_pi = self.pi(lstm_out)   # (N, L, n_gaussians)
        pi = F.softmax(logits_pi, -1)   # (N, L, n_g)
        
        batch_dim, seq_dim, _ = lstm_out.shape
        
        mu = self.mu(lstm_out)                                                                      # (N, L, n_g * l_dim)
        mu = mu.view(batch_dim, seq_dim, self.n_gaussians, self.latent_dim)                         # (N, L, n_g, l_dim)

        sigma = self.sigma(lstm_out)                                                                # (N, L, n_g * l_dim)
        sigma = torch.exp(sigma.view(batch_dim, seq_dim, self.n_gaussians, self.latent_dim)) + 1e-3 # (N, L, n_g, l_dim)

        return pi, mu, sigma, hidden
    
def gaussian_density(mu, sigma, y):
    """
    Compute per-component, per-timestep multivariate Gaussian density

    mu: [N, L, n_g, l_dim]
    sigma: [N, L, n_g, l_dim]
    y: [N, L, l_dim]

    returns (N, L, n_g) densities 

    """
    norm = 1.0 / torch.sqrt(torch.tensor(2.0) * torch.pi) 
    # y.shape     = [N, L, latent_dim] -> [N, L, 1, l_dim]
    y = y.unsqueeze(2)
    # mu.shape    = [N, L, n_g, l_dim]
    # sigma.shape = [N, L, n_g, l_dim]
    per_dim = torch.exp(-0.5 * ((y-mu)/sigma)**2) / (sigma) * norm # [N, L, n_g, l_dim]

    # product over l_dim to get the full latent dimension density
    # p(y | m_k, s_k)
    return torch.prod(per_dim, dim=-1) # (N, L, n_g)

def mdn_loss(pi, mu, sigma, y): 
    """
    pi: [N, L, n_g]
    """
    res = pi * gaussian_density(mu, sigma, y) # [N, L, n_g]
    res = torch.sum(res, -1) # P(y) - [N, L]
    res = -torch.log(res) # -log(P(y)) # [N, L]
    return torch.mean(res)