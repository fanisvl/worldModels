import sys
sys.path.append('.')
import os
import wandb
from modules.controller import Controller, CMA_ES, get_flat_params, set_params_from_flat
from modules.mdn_rnn import MDN_RNN
from modules.vae import VAE
import gymnasium as gym
from vizdoom import gymnasium_wrapper
from tqdm import tqdm
import time
import torch
import cv2
import numpy as np
import torch
from modules.dream_env import DreamEnv
import multiprocessing as mp
from functools import partial

def preprocess_obs(obs):
    """Resizes and normalizes an observation frame for VAE input."""
    resized = cv2.resize(obs['screen'], (64, 64))
    normalized = resized.astype(np.float32) / 255.0
    tensor_obs = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)
    return tensor_obs

def validate_in_real_env(controller, config, n_rollouts=100, is_random=False):
    """
    Validates a controller (or random policy) in the real VizDoom environment.
    Returns the average survival time over n_rollouts.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n--- Starting validation in real environment on {device} ({'Random Policy' if is_random else 'Best Controller'}) ---")

    # Load VAE and RNN for state updates
    vae = VAE(config['latent_dim'], inverted_colors=config['vae_inverted_colors'])
    vae.load_state_dict(torch.load(config['vae_path'], map_location=device))
    vae.to(device).eval()

    rnn = MDN_RNN(config['latent_dim'], config['action_dim'], config['hidden_dim'], config['n_layers'], config['n_gaussians'])
    rnn.load_state_dict(torch.load(config['rnn_path'], map_location=device))
    rnn.to(device).eval()

    if not is_random:
        controller.to(device).eval()

    real_env = gym.make("VizdoomTakeCover-v0")
    total_reward = 0.0

    for i in tqdm(range(n_rollouts), desc=f"Validating ({'Random' if is_random else 'Controller'})"):
        obs_dict, _ = real_env.reset()
        rnn_hidden = rnn.initial_hidden(1, device)
        terminated, truncated = False, False
        episode_reward = 0

        while not terminated and not truncated:
            if is_random:
                action = real_env.action_space.sample()
            else:
                preprocessed_obs = preprocess_obs(obs_dict).to(device)
                with torch.no_grad():
                    z = vae.encode(preprocessed_obs)
                    z = z.squeeze(0)

                    h = rnn_hidden[0][-1].squeeze(0)
                    c = rnn_hidden[1][-1].squeeze(0)
                    controller_input = torch.cat([z, h, c])
                    
                    action_logits = controller(controller_input)
                    action_cont = torch.tanh(action_logits)

                    if action_cont < -0.33: action = 2
                    elif action_cont > 0.33: action = 1
                    else: action = 0

            obs_dict, reward, terminated, truncated, _ = real_env.step(action)
            episode_reward += reward

            if not is_random:
                # Update RNN state even though we don't use its output for dreaming
                preprocessed_obs = preprocess_obs(obs_dict).to(device)
                with torch.no_grad():
                    z = vae.encode(preprocessed_obs) 
                    action_tensor = torch.tensor([action], device=device)
                    # print(f'z shape: {z.shape}, action shape: {action_tensor.shape}')
                    rnn_input = torch.cat([z.squeeze(), action_tensor], dim=-1).unsqueeze(0).unsqueeze(0)
                    _, _, _, _, rnn_hidden = rnn(rnn_input, rnn_hidden)
        
        total_reward += episode_reward

    real_env.close()
    avg_reward = total_reward / n_rollouts
    print(f"--- Validation finished. Average survival time: {avg_reward:.2f} ---")
    return avg_reward

def evaluate_fitness(env, controller, n_rollouts):
    """"
    To evaluate, the agent performs the task 16 times with differeent initial random seeds.
    The agent fitness value is the average cumulative reward of the 16 random rollouts.
    The cumulative reward is defined to be the number of time steps the agent manages to stay alive during a rollout.

    Each rollout in the environment runs for a maximum of 2100 time steps (60 seconds), 
    The task is considered solved if the average survival time over 100 consecutive rollouts is greater than 750 time steps (20 seconds)
    """
    total_reward = 0.0

    for i in range(n_rollouts):
        obs, _ = env.reset()
        terminated, truncated = False, False

        while not terminated and not truncated:
            z = torch.from_numpy(obs).to(env.device).flatten()
            h = env.rnn_hidden[0][-1].squeeze(0) # h from (h, c) tuple, last layer, remove batch dim
            c = env.rnn_hidden[1][-1].squeeze(0)
            controller_input = torch.cat([z, h, c])

            with torch.no_grad():
                action_logits = controller(controller_input)
                action_cont = torch.tanh(action_logits)  # [-1,1]
            
            # Map continuous action to discrete [-1, 0, 1]
            # RNN expects: {0: "None", 1: "Right", 2: "Left"}
            if action_cont < -0.33:
                action = 2
            elif action_cont > 0.33:
                action = 1
            else:
                action = 0
            
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
    
    return total_reward / n_rollouts

def evaluate_controller_params(params_vector, config):
    """Worker function to evaluate a single controller parameter vector."""
    # Force CPU usage in worker processes
    torch.set_num_threads(1)  # Prevent thread oversubscription
    device = 'cpu'
    
    # Recreate models in worker process
    rnn = MDN_RNN(config['latent_dim'], config['action_dim'], config['hidden_dim'], 
                  config['n_layers'], config['n_gaussians'])
    rnn.load_state_dict(torch.load(config['rnn_path'], map_location=device))
    rnn.to(device)
    rnn.eval()
    
    vae = VAE(config['latent_dim'], inverted_colors=config['vae_inverted_colors'])
    vae.load_state_dict(torch.load(config['vae_path'], map_location=device))
    vae.to(device)
    vae.eval()
    
    # Create controller and set parameters
    controller = Controller(config['latent_dim'], config['hidden_dim'], config['action_dim'])
    set_params_from_flat(controller, params_vector)
    controller.to(device)
    controller.eval()
    
    # Create dream environment
    # real_env_for_reset = gym.make("VizdoomTakeCover-v0")
    dream_env = DreamEnv(
        vae,
        rnn,
        None,
        latent_dim=config['latent_dim'],
        action_dim=3,
        render_mode='rgb_array',
        temperature=config['temperature'],
        max_episode_length=config['max_episode_length'],
    )
    
    try:
        fitness = evaluate_fitness(dream_env, controller, n_rollouts=config['n_eval_rollouts'])
        return fitness
    finally:
        dream_env.close()

if __name__ == '__main__':
    DEVICE = 'cpu'  # Force CPU for multiprocessing

    # --
    MAX_EPISODE_LENGTH = 2100

    # (V) and (M) parameters
    LATENT_DIM = 64
    VAE_INVERTED_COLORS = True
    HIDDEN_DIM=512
    ACTION_DIM = 1
    N_LAYERS = 1
    N_GAUSSIANS = 5
    TEMPERATURE = 1.15
    
    # CMA-ES
    POP_SIZE = 64
    N_GENERATIONS = 200
    N_EVAL_ROLLOUTS = 16
    SIGMA_INIT = 0.5
    WEIGHT_DECAY = 0.01
    VALIDATION_INTERVAL = 10
    REAL_VAL_ROLLOUTS = 100
    
    # Multiprocessing
    N_PROCESSES = mp.cpu_count() - 1  # Leave one core free
    print(f"Using {N_PROCESSES} processes for parallel evaluation")

    # Model Paths
    RNN_PATH = 'models/rnn/v1-17-08/rnn.lat64.nl.1.h512.seq300.e100.bs128.latent64_vizdoom_1_32_2.5k.17-08.cpt.10.pt'
    VAE_PATH = 'models/vae/vae.lat64.e100.bs128.inv.vizdoom_1_32_2.5k.14-08_epoch25.pt'
    BEST_CONTROLLER_DIR = 'checkpoints'
    CHECKPOINT_DIR = 'checkpoints'
    os.makedirs(BEST_CONTROLLER_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    # --

    # Create config dict for worker processes
    config = {
        'latent_dim': LATENT_DIM,
        'action_dim': ACTION_DIM,
        'hidden_dim': HIDDEN_DIM,
        'n_layers': N_LAYERS,
        'n_gaussians': N_GAUSSIANS,
        'vae_inverted_colors': VAE_INVERTED_COLORS,
        'temperature': TEMPERATURE,
        'max_episode_length': MAX_EPISODE_LENGTH,
        'n_eval_rollouts': N_EVAL_ROLLOUTS,
        'real_val_rollouts': REAL_VAL_ROLLOUTS,
        'rnn_path': RNN_PATH,
        'vae_path': VAE_PATH,
        # CMA-ES params
        'pop_size': POP_SIZE,
        'n_generations': N_GENERATIONS,
        'sigma_init': SIGMA_INIT,
        'weight_decay': WEIGHT_DECAY,
    }

    # Initialize wandb
    run_name = f"c_pop{POP_SIZE}_gen{N_GENERATIONS}_sigma{SIGMA_INIT}_wd{WEIGHT_DECAY}"
    wandb.init(
        project="dream_controller",
        config=config,
        name=run_name,
        mode="online",
    )

    # --- Baseline Validation ---
    random_policy_score = validate_in_real_env(None, config, n_rollouts=REAL_VAL_ROLLOUTS, is_random=True)
    wandb.log({"generation/random_policy_baseline": random_policy_score})


    temp_controller = Controller(LATENT_DIM, HIDDEN_DIM, ACTION_DIM)
    n_params = len(get_flat_params(temp_controller))
    print(f'Number of params in controller: {n_params}')
    cmaes = CMA_ES(n_params=n_params, sigma_init=SIGMA_INIT, pop_size=POP_SIZE, weight_decay=WEIGHT_DECAY)

    # Track best validation score
    best_validation_score = -np.inf
    best_params_real = None

    # Create multiprocessing pool
    with mp.Pool(processes=N_PROCESSES) as pool:
        pbar = tqdm(range(N_GENERATIONS), desc="Generation")
        for gen in pbar:
            pbar.set_description(f"Generation {gen}/{N_GENERATIONS}: Evaluating {POP_SIZE} candidates...")
            solutions = cmaes.ask()
            
            # Evaluate all solutions in parallel
            start_time = time.time()
            
            # Use partial to pass config to worker function
            worker_func = partial(evaluate_controller_params, config=config)
            rewards = pool.map(worker_func, solutions)
            
            eval_time = time.time() - start_time
            mean_reward = np.mean(rewards)
            std_reward = np.std(rewards)
            min_reward = min(rewards)
            max_reward = max(rewards)

            print(f"Evaluation completed in {eval_time:.2f} seconds")
            print(f"Fitness stats - Min: {min_reward:.2f}, Max: {max_reward:.2f}, Mean: {mean_reward:.2f}, Std: {std_reward:.2f}")

            cmaes.tell(rewards)

            # log
            best_params, best_reward, _, _ = cmaes.result()
            print(f'Generation: {gen:03d}, Best Fitness (Avg Survival Steps): {best_reward:.2f}')

            # Log to wandb
            log_data = {
                "generation": gen,
                "reward/mean_reward": mean_reward,
                "reward/std_reward": std_reward,
                "reward/min_reward": min_reward,
                "reward/max_reward": max_reward,
                "reward/best_dream_reward": best_reward,
                "reward/evaluation_time": eval_time,
            }

            # --- Periodic Validation in Real Environment ---
            if gen % VALIDATION_INTERVAL == 0 or gen == N_GENERATIONS - 1:
                best_controller_for_validation = Controller(LATENT_DIM, HIDDEN_DIM, ACTION_DIM)
                set_params_from_flat(best_controller_for_validation, best_params)
                
                real_world_score = validate_in_real_env(best_controller_for_validation, config, n_rollouts=REAL_VAL_ROLLOUTS)
                log_data["reward/best_avg_score_real"] = real_world_score

                # Save checkpoint if validation score improves
                if real_world_score > best_validation_score:
                    best_validation_score = real_world_score
                    best_params_real = best_params.copy() # Save the best parameters
                    wandb.run.summary["best_real_world_score"] = best_validation_score
                    checkpoint_path = os.path.join(CHECKPOINT_DIR, f'controller_gen_{gen}_score_{real_world_score:.2f}.pt')
                    torch.save(best_controller_for_validation.state_dict(), checkpoint_path)
                    print(f"\nNew best validation score: {real_world_score:.2f}. Saved checkpoint to {checkpoint_path}\n")
            wandb.log(log_data)

    final_best_params, final_best_reward, _, _ = cmaes.result()

    print("\n--- Training Finished ---")
    print(f"Final Best Reward in Dream: {final_best_reward:.2f}")   
    print(f"Best Validation Score in Real Env: {best_validation_score:.2f}")
    
    # save best controller based on real environment validation
    if best_params_real is not None:
        best_controller = Controller(LATENT_DIM, HIDDEN_DIM, ACTION_DIM)
        set_params_from_flat(best_controller, best_params_real)
        best_controller_path = os.path.join(BEST_CONTROLLER_DIR, f'best_controller_real_{best_validation_score:.2f}.pt')
        torch.save(best_controller.state_dict(), best_controller_path)
        print(f'Saved best controller (by real score) to {best_controller_path}')

        # Save model as a wandb artifact
        artifact = wandb.Artifact(f'controller-{run_name}', type='model')
        artifact.add_file(best_controller_path)
        logged_artifact = wandb.log_artifact(artifact)
    else:
        print("No validation was performed, or no improvement was seen. Final best model not saved.")

    wandb.finish()