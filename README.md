World Models - Ha, Schmidhuber (2018) implementation


## 1. Data Collection

- collect 100 rollouts of max 1k timesteps each.
- each rollouts contains pairs of (observation, action, reward, terminal)


## 2. Variational Autoencoder

Main Idea:
The goal of the VAE is to compress 64x64x3 observations to a latent dimension z (32).
It's trained on the observations from the random rollouts collected.

### VAE Theory

### The Reparameterization Trick




## 3. MDN-RNN

Main Idea:
Predict the next observation given the current (observation, action) pair.
We use a Mixture Density output layer because multiple outcomes could be plausible. 
So instead of predicting a single next observation, we predict the probability distribution
over the next observation.

### 3a. RNN (LSTM)
An LSTM is used to encode history of previous states to a hidden state h, which should contain
the necessary information to predict the next observation.

### 3b. Mixture Density Network


## 4. Experiments

### do we have to train the RNN? 
another re-implementation (https://ctallec.github.io/world-models/) found that even an untrained MDN-RNN might be useful


### training in latent space
VizDoom experiment from original paper
