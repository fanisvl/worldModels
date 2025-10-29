## Implementation of World Models - Ha, Schmidhuber (2018)

In this paper, 'world model' refers to a compressed (latent) representation of the environment and its dynamics. The goal is to learn this world model and utilize it to train a policy entirely within it, without the need for expensive trials in the real environment. 

The WM consists of three main components. 

1. A Variational Autoencoder which compresses the pixel observation input (dim=64x64x3) into a latent representation (dim=64).
2. A Mixture Density RNN which models the environment dynamics in latent space by predicting the next latent state z_{t+1} given the current latent state z_t and an action a_t. A Mixture of Gaussians is used to account for the inherent ambiguity of future states, instead of simply predicting the next state deterministically which would yield the average of all possible future states.
3. A Controller, which is linear layer that maps the current latent state z_t and the RNN hidden state to an action.

## Variational Autoencoder

A vanilla autoencoder is a neural network trained to reconstruct its input. It consists of an encoder that compresses a high-dimensional input x into a lower-dimensional latent representation z, and a decoder that reconstructs 
x from z. Because of the bottleneck and the reconstruction objective, the model learns to retain in z the most relevant information needed to faithfully reconstruct x.

The problem with the vanilla autoencoder is that the learned latent space of z is arbitrary and unstructured, so it's hard to sample new points in latent space and the prediction of the next latent state by the RNN will be difficult later. 

- variational (N prior)

also explain posterior collapse


## Mixture Density RNN

"A Mixture of Gaussians is used to account for the inherent ambiguity of future states, instead of simply predicting the next state deterministically which would yield the average of all possible future states."
explain this a bit more

