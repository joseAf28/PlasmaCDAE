import numpy as np
import torch 
import torch.nn as nn
import math


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



class SinusoidalNoiseEmbedding(nn.Module):
    def __init__(self, embed_dim: int = 64, max_freq: float = 1e4):
        super().__init__()
        self.embed_dim = embed_dim
        half = embed_dim // 2
        # Logarithmically spaced frequencies
        self.freqs = nn.Parameter(
            torch.exp(torch.linspace(math.log(1.0), math.log(max_freq), half)),requires_grad=False
        )
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, noise_level: torch.Tensor):
        # noise_level: [batch_size, 1]
        args = noise_level * self.freqs.unsqueeze(0)  # [B, half]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [B, embed_dim]
        return self.proj(emb)



class ConditionalDenoisingAutoencoderV2(nn.Module):
    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        x_embed_dim: int = 16,
        hidden_dim: int = 104, # 128
        noise_embed_dim: int = 10,
    ):
        super().__init__()
        # Embed x once
        self.x_proj = nn.Sequential(
            nn.Linear(x_dim, x_embed_dim),
            nn.GELU(),
            nn.Linear(x_embed_dim, x_embed_dim)
        )
        # Sinusoidal noise embedding
        self.noise_emb = SinusoidalNoiseEmbedding(noise_embed_dim)

        # Initial projection of y_noisy
        self.y_proj = nn.Sequential(
            nn.Linear(y_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Decoder: predict corrections for y
        self.decoder = nn.Sequential(
            nn.LayerNorm(hidden_dim + x_embed_dim + noise_embed_dim),
            nn.Linear(hidden_dim + x_embed_dim + noise_embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, y_dim + x_dim)
        )

    def forward(self, x: torch.Tensor, y_noisy: torch.Tensor, noise_level: torch.Tensor):
        # Embeddings
        x_emb = self.x_proj(x)                        # [B, x_embed_dim]
        noise_emb = self.noise_emb(noise_level)       # [B, noise_embed_dim]
        h = self.y_proj(y_noisy)                      # [B, hidden_dim]
        
        # Decoder
        dec_in = torch.cat([h, x_emb, noise_emb], dim=-1)
        dec_out = self.decoder(dec_in)                # [B, x_dim + y_dim]
        
        return dec_out, h

