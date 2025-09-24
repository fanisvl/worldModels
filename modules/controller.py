import numpy as np
import torch.nn as nn
import cma
import torch

class Controller(nn.Module):
    def __init__(self, in_latent_dim, in_hidden_dim, out_action_dim):
        super(Controller, self).__init__()

        self.latent_dim = in_latent_dim
        self.hidden_dim = in_hidden_dim
        self.out_action_dim = out_action_dim

        # In: (latent + lstm_cell + lstm_hidden)
        # Out: 1
        self.c = nn.Linear(in_latent_dim + in_hidden_dim + in_hidden_dim, out_action_dim)
    
    def forward(self, x):
        return self.c(x)

class CMA_ES():
    """
    Wrapper around cma.CMAEvolutionStrategy
    """
    def __init__(self, n_params, sigma_init=0.1, pop_size=255, weight_decay=0.01):
        self.n_params = n_params
        self.sigma_init = sigma_init
        self.pop_size = pop_size
        self.weight_decay = weight_decay
        self.solutions = None

        self.es = cma.CMAEvolutionStrategy(self.n_params * [0],
                                            self.sigma_init,
                                            {'popsize': self.pop_size})
        
    def ask(self):
        "Generate a batch of candidate solutions from the current search distribution."
        self.solutions = np.array(self.es.ask())
        return self.solutions
    
    def tell(self, reward_table_result):
        reward_table = -np.array(reward_table_result) # convert minimizer to maximizer
        
        # weight decay
        if self.weight_decay > 0:
            l2_decay = compute_weight_decay(self.weight_decay, self.solutions)
            reward_table += l2_decay
        
        # give list of fitness results back to the ES
        self.es.tell(self.solutions, reward_table.tolist())

    def result(self):
        r = self.es.result
        # best params so far, best reward overall, curr reward, sigma
        return (r[0], -r[1], -r[1], r[6])

def compute_weight_decay(weight_decay, model_param_list):
    model_param_grid = np.array(model_param_list)
    return - weight_decay * np.mean(model_param_grid * model_param_grid, axis=1)

# Utilities to convert between model parameters and flat vectors
def get_flat_params(model):
    params = []
    for p in model.parameters():
        params.append(p.data.cpu().numpy().flatten())
    return np.concatenate(params)

def set_params_from_flat(model, flat_params_vector):
    current_pos = 0
    for p in model.parameters():
        shape = p.data.shape
        num_elements = p.data.numel()
        param_slice = flat_params_vector[current_pos : current_pos + num_elements]
        p.data = torch.from_numpy(param_slice).view(shape).float()
        current_pos += num_elements

def load_controller(model, path):
    model.load_state_dict(torch.load(path))