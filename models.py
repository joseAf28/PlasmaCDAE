import numpy as np
import torch 
import torch.nn as nn

class ConditionalDenoisingAutoencoder(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_dim=128, latent_dim=64, noise_embed_dim=16):
        super(ConditionalDenoisingAutoencoder, self).__init__()
        # Embedding for noise level
        self.noise_embed = nn.Sequential(
            nn.Linear(1, noise_embed_dim),
            nn.ReLU(),
            nn.Linear(noise_embed_dim, noise_embed_dim)
        )
        # Now input_dim = x_dim + y_dim + noise_embed_dim
        input_dim = x_dim + y_dim + noise_embed_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU()
        )
        
        # Decoder: recouple x and the noise embedding for reconstruction.
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + x_dim + noise_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, x_dim + y_dim)
        )
    
    def forward(self, x, y_noisy, noise_level):
        # Embed the noise level scalar to a higher-dimensional vector.
        noise_embedded = self.noise_embed(noise_level)  # shape: [batch_size, noise_embed_dim]
        
        inp = torch.cat([x, y_noisy, noise_embedded], dim=1)
        latent = self.encoder(inp)
        dec_inp = torch.cat([latent, x, noise_embedded], dim=1)
        out = self.decoder(dec_inp)
        return out, latent



class MappingNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MappingNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
