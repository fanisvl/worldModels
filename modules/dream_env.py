import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import cv2
import torch.nn.functional as F

class DreamEnv(gym.Env):
    """
    A Gymnasium environment that simulates the world using a VAE and MDN-RNN.
    This is the "dream" environment where the agent will be trained.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, vae, rnn, real_env, latent_dim=64, action_dim=1, render_mode=None, temperature=1.15, max_episode_length=2100):
        super(DreamEnv, self).__init__()

        self.vae = vae
        self.rnn = rnn
        self.device = next(self.vae.parameters()).device
        self.real_env = real_env
        self.latent_dim = latent_dim
        self.temperature = temperature
        self.max_episode_length = max_episode_length
        
        # action_map = {0: "None", 1: "Right", 2: "Left"}
        self.action_space = spaces.Discrete(action_dim)
        # The observation space is the VAE's latent vector z
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(latent_dim,), dtype=np.float32)

        # Internal state
        self.z = None
        self.rnn_hidden = None
        self.cumulative_reward = 0
        self.steps = 0
        self.done_prob = 0.0
        self.last_action = 0
        
        self.render_mode = render_mode
        self.window = None

    def _preprocess_obs(self, obs_rgb):
        """Preprocesses an observation frame from the real environment."""
        obs_rgb = obs_rgb['screen']
        resized = cv2.resize(obs_rgb, (64, 64))
        normalized = resized.astype(np.float32) / 255.0
        tensor_obs = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)
        return tensor_obs.to(self.device)

    def _sample_next_z(self, pi_logits, mu, sigma_logits):
        pi_logits, mu, sigma_logits = pi_logits.squeeze(), mu.squeeze(), sigma_logits.squeeze()

        sigma = F.elu(sigma_logits) + 1.0 + 1e-4

        pi_dist = torch.distributions.Categorical(logits=pi_logits / self.temperature)
        mixture_idx = pi_dist.sample()
        
        selected_mu = mu[mixture_idx]
        selected_sigma = sigma[mixture_idx]

        z_dist = torch.distributions.Normal(selected_mu, selected_sigma)
        return z_dist.sample()

    def ground(self, obs_rgb=None):
        """Resets the dream state to a new observation from the real environment or provided observation."""
        if obs_rgb is not None:
            # Ground from provided observation (current real env state)
            print("Grounding dream from provided observation...")
            with torch.no_grad():
                processed_obs = self._preprocess_obs(obs_rgb)
                self.z = self.vae.encode(processed_obs)
        # 
        else:
            print("Cannot ground: No real environment or observation provided.")
            return self.z.squeeze(0).cpu().numpy(), {}

        self.rnn_hidden = self.rnn.initial_hidden(batch_size=1)
        self.rnn_hidden = (self.rnn_hidden[0].to(self.device), self.rnn_hidden[1].to(self.device))
        self.steps = 0
        self.done_prob = 0.0
        self.last_action = 0
        self.cumulative_reward = 0
        
        info = {}
        return self.z.squeeze(0).cpu().numpy(), info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Option to start from a pure dream state without a real environment.
        with torch.no_grad():
            if self.real_env:
                # Ground the dream in a real initial observation
                initial_obs_rgb, _ = self.real_env.reset(seed=seed)
                processed_obs = self._preprocess_obs(initial_obs_rgb)
                self.z = self.vae.encode(processed_obs)
            else:
                # Start from a random point in the latent space (pure dream)
                self.z = torch.randn(1, self.latent_dim).to(self.device)

        # Initialize the RNN hidden state
        self.rnn_hidden = self.rnn.initial_hidden(batch_size=1)
        
        self.rnn_hidden = (self.rnn_hidden[0].to(self.device), self.rnn_hidden[1].to(self.device))
        
        self.cumulative_reward = 0
        self.steps = 0
        self.done_prob = 0.0
        self.last_action = 0

        info = {}
        return self.z.squeeze(0).cpu().numpy(), info

    def step(self, action):
        """
        Takes an action and returns the next state, reward, done, and info.
        """
        # action is: {0: "None", 1: "Right", 2: "Left"}
        action_tensor = torch.tensor([action], device=self.device)
        self.last_action = action

        with torch.no_grad():
            # Prepare RNN input: [z_t, a_t]
            # Shape: (seq_len=1, batch_size=1, input_size=latent_dim + action_dim)
            rnn_input = torch.cat([self.z.squeeze(), action_tensor], dim=-1).unsqueeze(0).unsqueeze(0)
            
            # Get the next state distribution from the RNN
            pi_logits, mu, sigma_logits, done_logits, next_hidden = self.rnn(rnn_input, self.rnn_hidden)
            
            # Update the hidden state for the next step
            self.rnn_hidden = next_hidden
            
            # Sample the next latent state z_t+1
            self.z = self._sample_next_z(pi_logits, mu, sigma_logits).unsqueeze(0)

            # Determine if the episode is terminated
            done_prob = torch.sigmoid(done_logits).item()
            self.done_prob = done_prob
            terminated = (done_prob > 0.5)

        # The reward in "Take Cover" is 1 for each step survived
        reward = 1.0
        if terminated:
            reward = 0.0

        self.cumulative_reward += reward
        self.steps += 1
        
        # Truncation condition (max episode length)
        truncated = self.steps >= self.max_episode_length

        info = {'done_prob': done_prob} 
        
        if self.render_mode == "human":
            self.render()

        return self.z.squeeze(0).cpu().numpy(), reward, terminated, truncated, info

    def render(self):
        """Renders the current dream state by decoding z with the VAE."""
        with torch.no_grad():
            img_tensor = self.vae.decode(self.z)
        
        # Convert to numpy array for rendering
        img = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        img = (img * 255).astype(np.uint8)
        
        # Resize for better visualization, e.g., to 640x640
        render_img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_NEAREST)
        
        # Add text overlays for stats
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        color = (0, 255, 0) # Green
        thickness = 1
        
        action_map = {0: "Stay", 1: "Right", 2: "Left"}
        action_str = f"Action: {action_map.get(self.last_action, 'Unknown')}"
        steps_str = f"RNN Steps: {self.steps}"
        done_str = f"Done Prob: {self.done_prob:.3f}"
        
        cv2.putText(render_img, action_str, (10, 20), font, scale, color, thickness, cv2.LINE_AA)
        cv2.putText(render_img, steps_str, (10, 45), font, scale, color, thickness, cv2.LINE_AA)
        cv2.putText(render_img, done_str, (10, 70), font, scale, color, thickness, cv2.LINE_AA)

        if self.render_mode == "human":
            if self.window is None:
                cv2.namedWindow("Dream Environment", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Dream Environment", 640, 640)
                self.window = "Dream Environment" # Mark window as created
            
            cv2.imshow("Dream Environment", cv2.cvtColor(render_img, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
        
        return img
        
    def close(self):
        if self.window is not None:
            cv2.destroyAllWindows()
            self.window = None
        if self.real_env:
            self.real_env.close()