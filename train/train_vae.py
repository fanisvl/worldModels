
import torch
from torchvision import datasets, transforms
from modules.vae import VAE, vae_loss

# -- Load MNIST dataset --
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transforms = transforms.ToTensor()

# transforms = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5), (0.5))
# ])

mnist_data = datasets.MNIST(root='data', train=True, download=True, transform=transforms)
data_loader = torch.utils.data.DataLoader(
    dataset=mnist_data,
    batch_size=64,
    shuffle=True,
)

dataiter = iter(data_loader)
images, labels = next(dataiter)
print(torch.min(images), torch.max(images))

# -- Init VAE model --
model = VAE(latent_dim=16, in_shape=(1,28,28)).to(device)
opt = torch.optim.Adam(model.parameters(), lr = 1e-3, weight_decay=1e-5)

# -- Train VAE --
n_epochs = 10
outputs = []

for epoch in range(n_epochs):
    epoch_loss = 0.0
    num_batches = 0
    for i, (img, _) in enumerate(data_loader): # iterate batches
        # if i >= 10: # train on partial data for experiments
            # break
        img = img.to(device)

        recon_x, mu, log_var = model(img)
        loss = vae_loss(recon_x, img, mu, log_var)

        opt.zero_grad()
        loss.backward()
        opt.step()

        epoch_loss += loss.item()
        num_batches += 1

        # Store results from the first batch of each epoch for visualization
        if i == 0:
             outputs.append((epoch, img.detach().cpu(), recon_x.detach().cpu()))

    avg_epoch_loss = epoch_loss / (num_batches * data_loader.batch_size) # More representative loss
    print(f'Epoch: {epoch+1}, Average Loss: {avg_epoch_loss:.4f}')