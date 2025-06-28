import gymnasium as gym
import numpy as np
import os
import cv2
from tqdm import tqdm
import time

def generate_data(rollouts, max_timesteps, data_dir='recorded_data'):
    """
    Run several rollouts in the CarRacing-v3 environment.
    Each rollout stores:
        - observations: rendered RGB frames (3x96x96 by default, will resize to 3x64x64 to match the paper)
        - actions: actions taken at each frame
        - rewards: rewards per step
        - terminals: boolean flags for episode termination
        
    Args:
        rollouts: Number of episodes to run
        max_timesteps: Maximum timesteps per episode
        data_dir: Directory to save the collected data
        
    Returns:
        None (saves data to disk)
    """
    os.makedirs(data_dir, exist_ok=True)
    
    env = gym.make("CarRacing-v3", render_mode="rgb_array", max_episode_steps=max_timesteps)
    
    start = time.time()
    # *the generated track is random every episode
    for rollout_idx in tqdm(range(rollouts), desc="Generating rollouts"):
        # rollout data
        observations = []
        actions = []
        rewards = []
        terminals = []
        
        observation, info = env.reset()
        prev_action = np.array([0.0, 0.0, 0.0])

        # Skip the initial zoom-in frames
        skip_zoom_frames = 50
        for _ in range(skip_zoom_frames):
            action = np.array([0.0, 0.0, 0.0])
            observation, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                observation, info = env.reset()

        # Run rollout for specified max timesteps or until terminated
        for t in range(max_timesteps):
            # remove the bottom black bar that shows controls
            cropped_obs = observation[0:84, :, :]
            # resize to 64x64
            resized_obs = cv2.resize(cropped_obs, (64, 64))
            observations.append(resized_obs)
            
            # random policy
            noise_scale = 0.5  # Controls smoothness - lower is smoother
            random_change = np.random.normal(0, noise_scale, size=3)
            action = np.clip(prev_action + random_change, [-1.0, 0.0, 0.0], [1.0, 1.0, 1.0])
            prev_action = action
            actions.append(action)
            
            # action
            next_observation, reward, terminated, truncated, info = env.step(action)
            
            rewards.append(reward)
            terminals.append(terminated or truncated)
            
            observation = next_observation
            
            # Break if episode terminates
            if terminated or truncated:
                break
        
        # save
        episode_data = {
            'observations': np.array(observations),
            'actions': np.array(actions),
            'rewards': np.array(rewards),
            'terminals': np.array(terminals)
        }
        np.savez_compressed(
            os.path.join(data_dir, f'rollout_{rollout_idx:05d}.npz'),
            **episode_data
        )
    env.close()
    print(f"Generated {rollouts} rollouts with {max_timesteps} max timesteps each. \nSaved to {data_dir}")
    print(f"Time taken: {time.time() - start:.2f} seconds")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Generate CarRacing rollouts')
    parser.add_argument('--rollouts', type=int, required=True, help='Directory to save rollout data')
    parser.add_argument('--max_ts', type=int, required=True, help='Directory to save rollout data')
    parser.add_argument('--dir', type=str, default='saved_rollouts', help='Directory to save rollout data')
    args = parser.parse_args()
    generate_data(rollouts=100, max_timesteps=1000, data_dir=args.dir)