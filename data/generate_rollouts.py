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
        render_mode: Rendering mode (None for no rendering)
        data_dir: Directory to save the collected data
        
    Returns:
        None (saves data to disk)
    """
    os.makedirs(data_dir, exist_ok=True)
    
    env = gym.make("CarRacing-v3", render_mode="rgb_array")
    
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
        for t in tqdm(range(max_timesteps), desc='Steps'):
            # remove the bottom black bar that shows controls
            cropped_obs = observation[0:84, :, :]
            # resize from 96x96 to 64x64
            resized_obs = cv2.resize(cropped_obs, (64, 64))
            observations.append(resized_obs)
            
            # random policy
            noise_scale = 0.5  # Controls smoothness - lower is smoother
            random_change = np.random.normal(0, noise_scale, size=3)
            action = np.clip(prev_action + random_change, [-1.0, 0.0, 0.0], [1.0, 1.0, 1.0])
            prev_action = action
            actions.append(action)
            
            # Take action in environment
            next_observation, reward, terminated, truncated, info = env.step(action)
            
            rewards.append(reward)
            terminals.append(terminated or truncated)
            
            observation = next_observation
            
            # Break if episode terminates
            if terminated or truncated:
                break
        
        # Save the episode data
        episode_data = {
            'observations': np.array(observations),
            'actions': np.array(actions),
            'rewards': np.array(rewards),
            'terminals': np.array(terminals)
        }
        
        # Save to disk
        np.savez_compressed(
            os.path.join(data_dir, f'rollout_{rollout_idx:05d}.npz'),
            **episode_data
        )
    env.close()
    print(f"Generated {rollouts} rollouts, saved to {data_dir}")
    print(f"Time taken: {time.time() - start:.2f} seconds")
    

def inspect_data(data_dir='recorded_data', rollout_idx=0):
    """
    Inspect the generated rollout data from the CarRacing environment.
    
    Args:
        data_dir: Directory where rollout data is stored
        rollout_idx: Index of the specific rollout to inspect
        show_frames: If True, displays sampled frames from the rollout
        print_stats: If True, prints statistics about the rollout
        
    Returns:
        Dictionary containing the rollout data
    """
    import matplotlib.pyplot as plt
    from matplotlib import animation
    
    file_path = os.path.join(data_dir, f'rollout_{rollout_idx:05d}.npz')
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None

    data = np.load(file_path)
    observations = data['observations']
    actions = data['actions']
    rewards = data['rewards']
    terminals = data['terminals']
    
    # Print statistics
    print(f"Rollout #{rollout_idx} Statistics:")
    print(f"  Frames: {len(observations)}")
    print(f"  Observation shape: {observations.shape}")
    print(f"  Actions shape: {actions.shape}")
    print(f"  Total reward: {np.sum(rewards):.2f}")
    print(f"  Average reward per step: {np.mean(rewards):.4f}")
    print(f"  Terminated early: {any(terminals)}")
    if any(terminals):
        term_idx = np.where(terminals)[0][0]
        print(f"  Terminated at step: {term_idx}")
    
    # Display frames
    # Show first, middle and last frame
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    indices = [0, len(observations)//2, len(observations)-1]
    titles = ['First Frame', 'Middle Frame', 'Last Frame']
    for i, (idx, title) in enumerate(zip(indices, titles)):
        axes[i].imshow(observations[idx])
        axes[i].set_title(f"{title}\nAction: {actions[idx]}\nReward: {rewards[idx]:.2f}")
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()

    # Animation of rollout        
    fig, ax = plt.subplots(figsize=(8, 8))
    def init():
        ax.clear()
        return [ax]
    def animate(i):
        ax.clear()
        ax.imshow(observations[i])
        ax.set_title(f"Frame {i}/{len(observations)-1}, Action: {actions[i]}, Reward: {rewards[i]:.2f}")
        ax.axis('off')
        return [ax]
    anim = animation.FuncAnimation(fig, animate, init_func=init, 
                                    frames=len(observations), interval=50, blit=True)
    plt.tight_layout()
    plt.show()

    return {
        'observations': observations,
        'actions': actions,
        'rewards': rewards,
        'terminals': terminals
    }

if __name__ == "__main__":
    generate_data(rollouts=100, max_timesteps=6000, data_dir='saved_rollouts')
    # inspect_data(data_dir='saved_rollouts', rollout_idx=0)