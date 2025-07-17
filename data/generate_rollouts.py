import gymnasium as gym
from vizdoom import gymnasium_wrapper
import numpy as np
import os
import cv2
from tqdm import tqdm
import time

def generate_data(rollouts, data_dir='vizdoom/recorded_data'):
    """
    Run several rollouts in the VizdoomTakeCover-v0 environment.
    Each rollout stores:
        - observations: rendered RGB frames (3x96x96 by default, will resize to 3x64x64 to match the paper)
        - actions: actions taken at each frame
        - rewards: rewards per step
        - terminals: boolean flags for episode termination
        
    Args:
        rollouts: Number of episodes to run
        data_dir: Directory to save the collected data
        
    Returns:
        None (saves data to disk)
    """
    os.makedirs(data_dir, exist_ok=True)

    env = gym.make("VizdoomTakeCover-v0", render_mode='rgb_array')
    
    start = time.time()
    for rollout_idx in tqdm(range(rollouts), desc="Generating rollouts"):
        # rollout data
        observations = []
        actions = []
        rewards = []
        terminals = []
        
        observation, _ = env.reset()
        observation = observation['screen']

        # Run rollout for specified max timesteps or until terminated
        terminated, truncated = False, False
        while not terminated and not truncated:
            # resize to 64x64
            resized_obs = cv2.resize(observation, (64, 64))
            observations.append(resized_obs)
            
            # random policy
            action = np.random.choice([0, 1, 2])
            actions.append(action)
            next_observation, reward, terminated, truncated, info = env.step(action)
            next_observation = next_observation['screen']
            
            rewards.append(reward)
            terminals.append(terminated or truncated)
            
            observation = next_observation
        
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
    print(f"Generated {rollouts} rollouts. \nSaved to {data_dir}")
    print(f"Time taken: {time.time() - start:.2f} seconds")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Generate VizdoomTakeCover-v0 rollouts')
    parser.add_argument('--rollouts', type=int, default=200, help='Directory to save rollout data')
    parser.add_argument('--dir', type=str, default='vizdoom_rollouts/recorded_data', help='Directory to save rollout data')
    args = parser.parse_args()
    generate_data(rollouts=args.rollouts, data_dir=args.dir)