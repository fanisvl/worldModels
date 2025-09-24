import sys
import os
sys.path.append('.')
from modules.mdn_rnn import MDN_RNN
from modules.vae import VAE
from modules.dream_env import DreamEnv
from modules.controller import Controller, load_controller
import gymnasium as gym
from vizdoom import gymnasium_wrapper
import torch
import cv2
import time
import numpy as np

def preprocess_real_obs(obs_rgb):
    """Preprocess real environment observation for display."""
    obs_rgb = obs_rgb['screen']
    # Resize to match dream environment display size
    resized = cv2.resize(obs_rgb, (640, 640), interpolation=cv2.INTER_NEAREST)
    return resized

def create_side_by_side_display(real_img, dream_img, real_info, dream_info):
    """Create a side-by-side comparison of real and dream environments."""
    # Ensure both images are the same size
    height, width = 640, 640
    real_img = cv2.resize(real_img, (width, height), interpolation=cv2.INTER_NEAREST)
    dream_img = cv2.resize(dream_img, (width, height), interpolation=cv2.INTER_NEAREST)
    
    # Create combined image
    combined = np.hstack([real_img, dream_img])
    
    # Add labels and info
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.8
    color = (255, 255, 255)  # White
    thickness = 2
    
    # Labels
    cv2.putText(combined, "REAL ENVIRONMENT", (10, 30), font, scale, color, thickness, cv2.LINE_AA)
    cv2.putText(combined, "DREAM ENVIRONMENT", (width + 10, 30), font, scale, color, thickness, cv2.LINE_AA)
    
    # Control Info
    control_text = f"Control: {real_info.get('control_by', 'N/A')}"
    cv2.putText(combined, control_text, (10, 60), font, 0.6, color, 1, cv2.LINE_AA)

    # Real environment info
    cv2.putText(combined, f"Steps: {real_info['steps']}", (10, height - 60), font, 0.6, color, 1, cv2.LINE_AA)
    cv2.putText(combined, f"Action: {real_info['action']}", (10, height - 40), font, 0.6, color, 1, cv2.LINE_AA)
    cv2.putText(combined, f"Reward: {real_info['reward']:.1f}", (10, height - 20), font, 0.6, color, 1, cv2.LINE_AA)
    
    # Dream environment info  
    cv2.putText(combined, f"RNN Steps: {dream_info['steps']}", (width + 10, height - 80), font, 0.6, color, 1, cv2.LINE_AA)
    cv2.putText(combined, f"Action: {dream_info['action']}", (width + 10, height - 60), font, 0.6, color, 1, cv2.LINE_AA)
    cv2.putText(combined, f"Done Prob: {dream_info['done_prob']:.3f}", (width + 10, height - 40), font, 0.6, color, 1, cv2.LINE_AA)
    cv2.putText(combined, f"Reward: {dream_info['reward']:.1f}", (width + 10, height - 20), font, 0.6, color, 1, cv2.LINE_AA)
    
    return combined

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
    TEMPERATURE = 1.0
    MAX_TIMESTEPS = 2100
    
    # --- New Configuration ---
    # Set to False to start in a pure dream without a real environment
    GROUND_IN_REALITY = True 
    SHOW_COMPARISON = True  # Show side-by-side comparison
    USE_CONTROLLER = True # Set to True to use the controller
    CONTROLLER_PATH = 'models/controller_checkpoints/controller_gen_20_score_768.80.pt'

    RNN_PATH = 'models/rnn/v1-17-08/rnn.lat64.nl.1.h512.seq300.e100.bs128.latent64_vizdoom_1_32_2.5k.17-08.cpt.10.pt'
    VAE_PATH = 'models/vae/vae.lat64.e100.bs128.inv.vizdoom_1_32_2.5k.14-08_epoch25.pt'
    
    # --- Load Models ---
    print(f"Loading models to {DEVICE}...")
    rnn = MDN_RNN(LATENT_DIM, ACTION_DIM_RNN, HIDDEN_DIM, N_LAYERS, N_GAUSSIANS).to(DEVICE).eval()
    rnn.load_state_dict(torch.load(RNN_PATH, map_location=DEVICE))

    vae = VAE(LATENT_DIM, inverted_colors=VAE_INVERTED_COLORS).to(DEVICE).eval()
    vae.load_state_dict(torch.load(VAE_PATH, map_location=DEVICE))

    controller = None
    if USE_CONTROLLER:
        try:
            controller = Controller(LATENT_DIM, HIDDEN_DIM, ACTION_DIM_RNN).to(DEVICE).eval()
            load_controller(controller, CONTROLLER_PATH)
            print("Controller loaded successfully.")
        except FileNotFoundError:
            print(f"Controller file not found at {CONTROLLER_PATH}. Controller will be disabled.")
            USE_CONTROLLER = False
            controller = None

    print("Models loaded successfully.")

    # --- Setup Environments ---
    print("Setting up environments...")
    
    # Real environment for grounding and comparison
    real_env = None
    if GROUND_IN_REALITY:
        print("Creating real environment instance.")
        real_env = gym.make("VizdoomTakeCover-v0")
    
    # Dream environment (without render_mode if we're doing custom rendering)
    render_mode = None if (GROUND_IN_REALITY and SHOW_COMPARISON) else 'human'
    dream_env = DreamEnv(
        vae,
        rnn,
        real_env,
        latent_dim=LATENT_DIM,
        render_mode=render_mode,
        temperature=TEMPERATURE,
        max_episode_length=MAX_TIMESTEPS,
    )
    
    # --- Initialize environments ---
    dream_obs, dream_info = dream_env.reset()
    
    real_obs = None
    real_steps = 0
    real_reward = 0
    real_terminated = False
    real_truncated = False
    
    if real_env:
        real_obs, _ = real_env.reset()
        
    # Setup display window
    if GROUND_IN_REALITY and SHOW_COMPARISON:
        cv2.namedWindow("Real vs Dream Comparison", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Real vs Dream Comparison", 1280, 640)
    
    running = True
    
    print("\n--- Starting Visualization ---")
    print("Controls:")
    print("  'a' -> Move Left")
    print("  'd' -> Move Right")
    if GROUND_IN_REALITY:
        print("  'g' -> Ground Dream in Reality (reset both environments)")
    if USE_CONTROLLER:
        print("  'c' -> Toggle Controller ON/OFF")
    print("  'q' or ESC -> Quit")
    
    action_map = {0: "Stay", 1: "Right", 2: "Left"}
    last_action = 0
    controller_active = USE_CONTROLLER
    
    while running:
        start_time = time.time()
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == 27:  # 'q' or ESC key
            running = False
            continue

        # Default action is to stay
        action = 0
        control_by = "User" # Assume user control by default for this frame
        
        # Update action based on key press
        if key == ord('a'):
            action = 2  # Left
            controller_active = False # User override disables controller for this action
        elif key == ord('d'):
            action = 1  # Right
            controller_active = False # User override disables controller for this action
        elif key == ord('c') and USE_CONTROLLER:
            controller_active = not controller_active
            print(f"Controller is now {'ACTIVE' if controller_active else 'INACTIVE'}")
        elif key == ord('g') and GROUND_IN_REALITY:
            # Ground the dream in the current real environment state (don't reset real env)
            print("Grounding dream in current reality state...")
            if real_env and real_obs is not None:
                # Ground the dream using the current real observation
                dream_obs, dream_info = dream_env.ground(obs_rgb=real_obs)
            last_action = 0
            continue
        elif controller_active and controller:
            control_by = "Controller"
            with torch.no_grad():
                z = torch.from_numpy(dream_obs).unsqueeze(0).to(DEVICE)
                h, c = dream_env.rnn_hidden
                h, c = h.to(DEVICE), c.to(DEVICE)
                
                controller_input = torch.cat([z, h.squeeze(0), c.squeeze(0)], dim=1)
                action_val = controller(controller_input).item()
                
                # Discretize controller output
                if action_val < -0.5:
                    action = 2 # Left
                elif action_val > 0.5:
                    action = 1 # Right
                else:
                    action = 0 # Stay
        else:
            # No key pressed and controller is off, so action remains 0 (Stay)
            # control_by is already "User"
            pass
            
        last_action = action
        
        # Step both environments with the same action
        dream_obs, dream_reward, dream_terminated, dream_truncated, dream_info = dream_env.step(action)
        
        if real_env and not (real_terminated or real_truncated):
            real_obs, real_reward, real_terminated, real_truncated, _ = real_env.step(action)
            real_steps += 1
        
        # Handle environment resets
        if dream_terminated or dream_truncated:
            reason = "terminated (model predicted done)" if dream_terminated else "truncated (max steps)"
            print(f"Dream episode ended: {reason}. Resetting...")
            dream_obs, dream_info = dream_env.reset()
            
        if real_env and (real_terminated or real_truncated):
            reason = "terminated" if real_terminated else "truncated"
            print(f"Real episode ended: {reason}. Resetting...")
            real_obs, _ = real_env.reset()
            real_steps = 0
            real_reward = 0
            real_terminated = False
            real_truncated = False
        
        # Render comparison or individual views
        if GROUND_IN_REALITY and SHOW_COMPARISON and real_env:
            # Get dream environment rendered image
            dream_img = dream_env.render()
            dream_img_bgr = cv2.cvtColor(dream_img, cv2.COLOR_RGB2BGR)
            
            # Preprocess real environment image
            real_img = preprocess_real_obs(real_obs)
            real_img_bgr = cv2.cvtColor(real_img, cv2.COLOR_RGB2BGR)
            
            # Create info dictionaries
            real_info = {
                'steps': real_steps,
                'action': action_map.get(last_action, 'Unknown'),
                'reward': real_reward,
                'control_by': control_by
            }
            
            dream_info_display = {
                'steps': dream_env.steps,
                'action': action_map.get(last_action, 'Unknown'),
                'done_prob': dream_info.get('done_prob', 0.0),
                'reward': dream_reward
            }
            
            # Create and display combined view
            combined_img = create_side_by_side_display(real_img_bgr, dream_img_bgr, real_info, dream_info_display)
            cv2.imshow("Real vs Dream Comparison", combined_img)
            
        elif not SHOW_COMPARISON:
            # Let dream environment handle its own rendering
            pass
    
        # Ensure consistent timing (100ms per step)
        elapsed = time.time() - start_time
        sleep_time = max(0, 0.1 - elapsed)
        time.sleep(sleep_time)
            
    # Cleanup
    dream_env.close()
    if real_env:
        real_env.close()
    cv2.destroyAllWindows()
    print("Visualization finished.")