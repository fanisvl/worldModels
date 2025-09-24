import sys
import os
sys.path.append('.')
from modules.mdn_rnn import MDN_RNN
from modules.vae import VAE
from modules.dream_env import DreamEnv
import gymnasium as gym
from vizdoom import gymnasium_wrapper
import torch
import cv2
import time

if __name__ == '__main__':
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --- Configuration ---
    # (V) and (M) parameters
    LATENT_DIM = 64
    VAE_INVERTED_COLORS = True
    HIDDEN_DIM = 512
    ACTION_DIM_RNN = 1
    N_LAYERS = 1
    N_GAUSSIANS = 5
    TEMPERATURE = 1.25
    MAX_TIMESTEPS = 2100
    # allows the usage of a real environment to optionally ground the dream
    GROUND_IN_REALITY = False

    RNN_PATH = 'models/rnn/v1-17-08/rnn.lat64.nl.1.h512.seq300.e100.bs128.latent64_vizdoom_1_32_2.5k.17-08.cpt.10.pt'
    VAE_PATH = 'models/vae/vae.lat64.e100.bs128.inv.vizdoom_1_32_2.5k.14-08_epoch25.pt'
    
    # --- Load Models ---
    print(f"Loading models to {DEVICE}...")
    rnn = MDN_RNN(LATENT_DIM, ACTION_DIM_RNN, HIDDEN_DIM, N_LAYERS, N_GAUSSIANS).to(DEVICE).eval()
    rnn.load_state_dict(torch.load(RNN_PATH, map_location=DEVICE))

    vae = VAE(LATENT_DIM, inverted_colors=VAE_INVERTED_COLORS).to(DEVICE).eval()
    vae.load_state_dict(torch.load(VAE_PATH, map_location=DEVICE))
    print("Models loaded successfully.")

    # --- Setup Environment ---
    print("Setting up Dream Environment...")
    real_env_for_reset = None
    if GROUND_IN_REALITY:
        print("Creating a real environment instance for grounding.")
        real_env_for_reset = gym.make("VizdoomTakeCover-v0")
    else:
        print("Starting in pure dream mode. No real environment used.")
    dream_env = DreamEnv(
        vae,
        rnn,
        real_env_for_reset,
        latent_dim=LATENT_DIM,
        render_mode='human',
        temperature=TEMPERATURE,
        max_episode_length=MAX_TIMESTEPS,
    )
    
    # --- Main Visualization Loop ---
    obs, info = dream_env.reset()
    running = True
    
    print("\n--- Starting Dream Visualization ---")
    print("Controls:")
    print("  'a' -> Move Left")
    print("  'd' -> Move Right")
    if GROUND_IN_REALITY:
        print("  'g' -> Ground in Reality (reset to real frame)")
    print("  'q' or ESC -> Quit")
    
    action_map = {0: "Stay", 1: "Right", 2: "Left"}
    while running:
        start_time = time.time()
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27: # 'q' or ESC key
            running = False
            continue

        action = 0 
        if key == ord('a'):
            action = 2 # Left
        elif key == ord('d'):
            action = 1 # Right
        elif key == ord('g') and GROUND_IN_REALITY:
            obs, info = dream_env.ground()
            action = 0 # Reset action after grounding
            dream_env.render()
            time.sleep(0.1)
            continue
            
        # Step the environment with the chosen action
        obs, reward, terminated, truncated, info = dream_env.step(action)
        if terminated or truncated:
            reason = "terminated (model predicted done)" if terminated else "truncated (max steps)"
            print(f"Dream episode ended: {reason}. Resetting...")
            obs, info = dream_env.reset()
        elapsed = time.time() - start_time
        sleep_time = max(0, 0.1 - elapsed)
        time.sleep(sleep_time)
    dream_env.close()
    print("Visualization finished.")