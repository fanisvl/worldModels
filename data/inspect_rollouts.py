import numpy as np
import os

def inspect_data(data_dir, rollout_idx):
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

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Inspect CarRacing rollouts')
    parser.add_argument('--dir', type=str, required=True, help='Directory of the rollout data')
    parser.add_argument('--idx', type=int, required=True, help='Index of the rollout to inspect')
    args = parser.parse_args()
    inspect_data(data_dir=args.dir, rollout_idx=args.idx)