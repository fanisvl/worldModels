import torch
import numpy as np
import cv2
import time
import sys
sys.path.append('.')
from modules.vae import VAE
from modules.mdn_rnn import MDN_RNN

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VAE_PATH = 'models/vae/vae.lat64.e100.bs128.inv.vizdoom_1_32_2.5k.14-08_epoch25.pt'
MDN_RNN_PATH = 'models/rnn/v1-17-08/rnn.lat64.nl.1.h512.seq300.e100.bs128.latent64_vizdoom_1_32_2.5k.17-08.cpt.30.pt'
LATENT_DIM = 64
ACTION_DIM = 1
HIDDEN_SIZE = 512
N_GAUSSIANS = 5
NUM_LAYERS = 1
SEQ_LEN = 300

TEMPERATURE = 1.0

# Global action: single integer for VizDoom [0: move_left, 1: move_right, 2: shoot]
action = 0 # Default action
running = True
key_pressed = False

def handle_keyboard_cv2():
    """Handle keyboard input for discrete actions using OpenCV's waitKey."""
    global action, running, key_pressed
    
    key = cv2.waitKey(20) & 0xFF # Increased wait time for better key capture
    
    key_pressed = False
    if key == ord('q') or key == 27:  # 'q' or ESC
        running = False
    elif key == ord('a'):  # move left
        action = 2
        key_pressed = True
    elif key == ord('d'):  # move right
        action = 1
        key_pressed = True
    else:
        action = 0
        key_pressed = True

# --- Helper Functions ---
def sample_from_mdn(pi, mu, log_sigma, temperature=1.0):
    """Samples a latent vector from the MDN output with temperature."""
    pi = pi.squeeze()
    mu = mu.squeeze()
    sigma = torch.exp(log_sigma.squeeze())

    if temperature == 0: # Deterministic
        k = torch.argmax(pi) # choose the most likely mixture component
        z_next = mu[k]
    else: # Stochastic
        # Apply softmax with temperature to get probabilities
        pi_probs = torch.softmax(pi / temperature, dim=-1)
        mixture = torch.distributions.Categorical(probs=pi_probs)
        k = mixture.sample()

        mu_k = mu[k]
        sigma_k = sigma[k]

        z_next = torch.normal(mu_k, sigma_k * np.sqrt(temperature))

        return z_next.unsqueeze(0)

def postprocess_frame(frame_tensor):
    """Converts a tensor to a displayable numpy image."""
    frame = frame_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    frame = np.clip(frame, 0, 1) # Ensure values are in the valid range
    frame = (frame * 255.0).astype(np.uint8)
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

# --- Main Execution ---
if __name__ == "__main__":
    # Load models
    vae = VAE(LATENT_DIM, inverted_colors=True).to(DEVICE)
    vae.load_state_dict(torch.load(VAE_PATH, map_location=DEVICE))
    vae.eval()

    mdn_rnn = MDN_RNN(LATENT_DIM, ACTION_DIM, HIDDEN_SIZE, NUM_LAYERS, N_GAUSSIANS).to(DEVICE)
    mdn_rnn.load_state_dict(torch.load(MDN_RNN_PATH, map_location=DEVICE))
    mdn_rnn.eval()

    try:
        # Initialize dream state
        with torch.no_grad():
            z = torch.randn(1, LATENT_DIM).to(DEVICE)
            rnn_hidden = mdn_rnn.initial_hidden()
            rnn_hidden = (rnn_hidden[0].to(DEVICE), rnn_hidden[1].to(DEVICE))

        step_count = 0
        done_prob = 0.0

        print("Starting VizDoom dream. Use A (left), D (right).")
        print("Press 'q' or ESC to exit.")
        print(f"RNN will auto-reset every {SEQ_LEN} steps or if 'done' is predicted.")

        while running:
            handle_keyboard_cv2()
            
            # --- Dream Step ---
            with torch.no_grad():
                action_tensor = torch.tensor([action], dtype=torch.float32).to(DEVICE)
                
                rnn_input = torch.cat([z.squeeze(), action_tensor], dim=-1).unsqueeze(0).unsqueeze(0)
                pi, mu, sigma, done_logits, rnn_hidden = mdn_rnn(rnn_input, rnn_hidden)
                
                done_prob = torch.sigmoid(done_logits).item()
                
                z = sample_from_mdn(pi, mu, sigma, temperature=TEMPERATURE)
                
                dream_frame_tensor = vae.decode(z)
                dream_frame_display = postprocess_frame(dream_frame_tensor)
                dream_frame_display = cv2.resize(dream_frame_display, (640, 480), interpolation=cv2.INTER_NEAREST)

            step_count += 1
            # Reset if sequence length is reached or model predicts terminal state
            if step_count >= SEQ_LEN or done_prob > 0.5:
                rnn_hidden = mdn_rnn.initial_hidden()
                rnn_hidden = (rnn_hidden[0].to(DEVICE), rnn_hidden[1].to(DEVICE))
                reset_reason = "sequence length" if step_count >= SEQ_LEN else "predicted terminal"
                step_count = 0
                print(f"Auto-reset: RNN hidden state reset due to {reset_reason}.")


            # --- Visualization ---
            cv2.putText(dream_frame_display, f"Steps: {step_count}/{SEQ_LEN}", (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            action_map = {0: "None", 1: "Right", 2: "Left"}
            action_str = f"Action: {action_map.get(action, 'N/A')}"
            action_text_color = (0, 255, 255) if key_pressed else (255, 255, 255) # Highlight when key is pressed
            cv2.putText(dream_frame_display, action_str, (10, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.7, action_text_color, 2)

            done_str = f"Done Prob: {done_prob:.3f}"
            done_text_color = (0, 0, 255) if done_prob > 0.5 else (255, 255, 255)
            cv2.putText(dream_frame_display, done_str, (10, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.7, done_text_color, 2)
            
            cv2.imshow("Dream Environment", dream_frame_display)
            
            time.sleep(0.1) # Adjusted for VizDoom's typical framerate
    
    finally:
        print("Exiting.")
        cv2.destroyAllWindows()