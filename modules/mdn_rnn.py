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

    def initial_hidden(self):
        # initialize both h0 and c0: shape (num_layers, batch=1, hidden_dim)
        num_layers = self.num_layers   # or hard‐code 1/2
        h0 = torch.zeros(num_layers, 1, self.hidden_size)
        c0 = torch.zeros(num_layers, 1, self.hidden_size)
        rnn_hidden = (h0, c0)
        return rnn_hidden
    
def log_gaussian_density(mu, raw_sigma, y):
    """
    Compute per-component, per-timestep multivariate Gaussian density

    mu: [N, L, n_g, l_dim]
    sigma: [N, L, n_g, l_dim]
    y: [N, L, l_dim]

    returns (N, L, n_g) densities 

    """
    y = y.unsqueeze(2) # [N, L, latent_dim] -> [N, L, 1, l_dim]
    sigma = torch.exp(raw_sigma) + 1e-4

    # log gaussian
    log_prob_per_latent = (
        -torch.log(sigma)
        -0.5 * torch.log(torch.tensor(2.0 * torch.pi))
        -0.5 * ((y-mu) / sigma) ** 2
    )

    # Sum over latent dimensions (log(prod(p_i)) = sum(log(p_i)) to avoid the underflow from taking the product)
    # log p(y | m_k, s_k)
    return torch.sum(log_prob_per_latent, dim=-1) # [N, L, n_g]

def mdn_loss(pi_logits, mu, sigma, y): 
    """
    pi_logits: [N, L, n_g] - Raw logits for the mixture components
    mu: [N, L, n_g, l_dim]
    raw_sigma: [N, L, n_g, l_dim]
    y: [N, L, l_dim]
    """
    log_gaussian = log_gaussian_density(mu, sigma, y)
    log_pi = F.log_softmax(pi_logits, dim=-1)
    log_weighted = log_pi + log_gaussian # [N, L, n_g]
    # log(sum(pi*gaussian)), log-sum-exp helps with numericaly stability by subtracting the maximum value:
    # log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))
    log_prob = torch.logsumexp(log_weighted, dim=-1) # [N,L]
    return -torch.mean(log_prob)