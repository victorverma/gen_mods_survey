import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import make_moons
from sklearn.neighbors import KernelDensity
import ptflops

class VAE_moon_unlab(nn.Module):
    def __init__(self, hidden_dim, latent_dim, input_dim=2, output_dim=2):
        super(VAE_moon_unlab, self).__init__()
        
        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc31 = nn.Linear(hidden_dim, latent_dim)
        self.fc32 = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.fc4 = nn.Linear(latent_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, hidden_dim)
        self.fc7 = nn.Linear(hidden_dim, output_dim)
    
    def encode(self, x):
        h1 = torch.relu(self.fc1(x))
        h2 = torch.relu(self.fc2(h1))
        h3 = torch.relu(self.fc2(h2))
        mu = self.fc31(h3)
        logvar = self.fc32(h3)
        logvar = torch.clamp(logvar, min=-10, max=10)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mu)
        return mu + eps * std
    
    def decode(self, z):
        h4 = torch.relu(self.fc4(z))
        h5 = torch.relu(self.fc5(h4))
        h6 = torch.relu(self.fc6(h5))
        return self.fc7(h6)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def loss_fn(recon_x, x, mu, logvar, kl_weight=0.1):
    recon_loss = nn.MSELoss(reduction='sum')(recon_x, x)
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.clamp(min=-10, max=10).exp())
    
    return recon_loss + kl_weight * kl_div

def count_flops(model):
    flops, params = ptflops.get_model_complexity_info(model, (1, 2), as_strings=True, print_per_layer_stat=True)
    return flops, params

def kl_weight_scheduler(epoch, NUM_EPOCHS, KL_WEIGHT):
    if epoch/NUM_EPOCHS < 0.9:
        return KL_WEIGHT
    elif epoch/NUM_EPOCHS < 0.95:
        return KL_WEIGHT + 0.1
    else:
        return KL_WEIGHT + 0.2

def train(model, dataloader, optimizer, device, NUM_EPOCHS, PRINT_LAP, KL_WEIGHT):
    epoch_loss = []
    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        for batch_data in dataloader:
            model.train()
            x = batch_data[0].to(device)
            optimizer.zero_grad()
            
            recon_x, mu, logvar = model(x)
            loss = loss_fn(recon_x, x, mu, logvar, KL_WEIGHT)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item() * x.size(0)
        
        epoch_loss.append(running_loss / len(dataloader.dataset))
        if epoch % PRINT_LAP == 0:
            print(f'Epoch [{epoch+1}/{NUM_EPOCHS}] -> loss:{epoch_loss[-1]:.2f}')
    return model

def model_save(model, file_path):
    torch.save(model.state_dict(), file_path)
    print(f"Model saved to {file_path}")

def model_load(model_class, file_path, device="cpu"):
    model = model_class()
    model.load_state_dict(torch.load(file_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Model loaded from {file_path} to {device}")
    return model
