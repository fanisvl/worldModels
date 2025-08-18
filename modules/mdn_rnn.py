import torch
import torch.nn as nn
import torch.nn.functional as F

class MDN_RNN(nn.Module):
    def __init__(self, latent_dim, action_dim, hidden_size, num_layers, n_gaussians, dropout=0.0):
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
        
        self.dropout = nn.Dropout(dropout)
        self.pi = nn.Linear(hidden_size, n_gaussians)
        self.mu = nn.Linear(hidden_size, n_gaussians * latent_dim)
        self.sigma = nn.Linear(hidden_size, n_gaussians * latent_dim)
        self.done_predictor = nn.Linear(hidden_size, 1)

    def forward(self, x, hidden=None):
        lstm_out, hidden = self.lstm(x, hidden)      # (N, L, H_out), hidden state
        lstm_out = self.dropout(lstm_out)
        logits_pi = self.pi(lstm_out)   # (N, L, n_gaussians)
        
        batch_dim, seq_dim, _ = lstm_out.shape
        
        mu = self.mu(lstm_out)                                                                      # (N, L, n_g * l_dim)
        mu = mu.view(batch_dim, seq_dim, self.n_gaussians, self.latent_dim)                         # (N, L, n_g, l_dim)

        sigma_logits = self.sigma(lstm_out)                                                                # (N, L, n_g * l_dim)
        sigma_logits = sigma_logits.view(batch_dim, seq_dim, self.n_gaussians, self.latent_dim) + 1e-3 # (N, L, n_g, l_dim)

        done_logits = self.done_predictor(lstm_out).squeeze(-1) # (N, L)

        return logits_pi, mu, sigma_logits, done_logits, hidden

    def initial_hidden(self, batch_size, device=None):
        # initialize both h0 and c0: shape (num_layers, batch=1, hidden_dim)
        num_layers = self.num_layers   # or hard‐code 1/2
        if device is None:
            device = next(self.parameters()).device
        h0 = torch.zeros(num_layers, batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(num_layers, batch_size, self.hidden_size, device=device)
        rnn_hidden = (h0, c0)
        return rnn_hidden
    
def log_gaussian_density(mu, sigma_logits, y):
    """
    Compute per-component, per-timestep multivariate Gaussian density

    mu: [N, L, n_g, l_dim]
    sigma: [N, L, n_g, l_dim]
    y: [N, L, l_dim]

    returns (N, L, n_g) densities 

    """
    y = y.unsqueeze(2) # [N, L, latent_dim] -> [N, L, 1, l_dim]
    sigma = F.elu(sigma_logits) + 1.0 + 1e-4 # ELU is capped at -1, so this is always > 0

    # log gaussian
    log_prob_per_latent = (
        -torch.log(sigma)
        -0.5 * torch.log(torch.tensor(2.0 * torch.pi))
        -0.5 * ((y-mu) / sigma) ** 2
    )

    # Sum over latent dimensions (log(prod(p_i)) = sum(log(p_i)) to avoid the underflow from taking the product)
    # log p(y | m_k, s_k)
    return torch.sum(log_prob_per_latent, dim=-1) # [N, L, n_g]

def mdn_loss(pi_logits, mu, sigma_logits, y): 
    """
    pi_logits: [N, L, n_g] - Raw logits for the mixture components
    mu: [N, L, n_g, l_dim]
    sigma_logits: [N, L, n_g, l_dim]
    y: [N, L, l_dim]
    """
    log_gaussian = log_gaussian_density(mu, sigma_logits, y)
    log_pi = F.log_softmax(pi_logits, dim=-1)
    log_weighted = log_pi + log_gaussian # [N, L, n_g]
    # log(sum(pi*gaussian)), log-sum-exp helps with numericaly stability by subtracting the maximum value:
    # log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))
    log_prob = torch.logsumexp(log_weighted, dim=-1) # [N,L]
    return -torch.mean(log_prob)

def rnn_combined_loss(pi_logits, mu, sigma_logits, done_logits, targets, BCE_POS_WEIGHT, DONE_LOSS_WEIGHT=1.0):
    """
    pi_logits: [N, L, n_g] - Raw logits for the mixture components
    mu: [N, L, n_g, l_dim]
    sigma_logits: [N, L, n_g, l_dim]
    done_logits: [N, L]
    y: {'next_latent': [N, L, l_dim], 'is_terminal': [N, L, 1]}
    BCE_POS_WEIGHT: 
    """
    latent_target = targets['next_latent']
    mdn_l = mdn_loss(pi_logits, mu, sigma_logits, latent_target)
    
    terminal_target = targets['is_terminal'].squeeze(-1) # (N,L)
    bce_loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=BCE_POS_WEIGHT)
    terminal_l = DONE_LOSS_WEIGHT * bce_loss_fn(done_logits, terminal_target)
    combined_loss = mdn_l + terminal_l
    return combined_loss, mdn_l, terminal_l