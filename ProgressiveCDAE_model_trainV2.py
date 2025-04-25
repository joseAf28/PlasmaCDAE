import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import config as cfg
import math


class LTPDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


class SinusoidalNoiseEmbedding(nn.Module):
    def __init__(self, embed_dim: int = 64, max_freq: float = 1e4):
        super().__init__()
        self.embed_dim = embed_dim
        half = embed_dim // 2
        # Logarithmically spaced frequencies
        self.freqs = nn.Parameter(
            torch.exp(
                torch.linspace(
                    math.log(1.0), math.log(max_freq), half
                )
            ),
            requires_grad=False
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


class FactorizedLinear(nn.Module):
    
    def __init__(self, in_features: int, out_features: int, rank: int = 32, bias: bool = True):
        super().__init__()
        self.factor1 = nn.Linear(in_features, rank, bias=False)
        self.factor2 = nn.Linear(rank, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.factor2(self.factor1(x))



class ConditionalDenoisingAutoencoderV2(nn.Module):
    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        x_embed_dim: int = 16,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        noise_embed_dim: int = 10,
        factor_rank: int = 16
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
        dec_out = self.decoder(dec_in)  # [B, x_dim + y_dim]
        
        return dec_out, h



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


def train_system(seed):

    torch.manual_seed(seed)
    np.random.seed(seed)
        
    ###* Prepare data

    with open(cfg.config_set["data_path"]) as f:
        data = f.readlines()


    for i in range(len(data)):
        data[i] = data[i].split()
        data[i] = [float(x) for x in data[i]]
        
    data = np.array(data)
    ratio_test_val_train = cfg.config_set["ratio_test_val_train"]


    X_train, X_temp, y_train, y_temp = train_test_split(data[:, 0:3], data[:, 3:], test_size=ratio_test_val_train, random_state=seed+1)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.7, random_state=seed+1)


    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X_train)
    y_scaled = scaler_Y.fit_transform(y_train)

    batch_size = cfg.config_set["batch_size"]

    dataset = LTPDataset(X_scaled, y_scaled)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


    X_val_scaled = scaler_X.transform(X_val)
    y_val_scaled = scaler_Y.transform(y_val)

    val_dataset = LTPDataset(X_val_scaled, y_val_scaled)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    X_scaled_test = scaler_X.transform(X_test)
    y_scaled_test = scaler_Y.transform(y_test)

    test_dataset = LTPDataset(X_scaled_test, y_scaled_test)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



    ###* Define the CDAE  and train the model

    x_dim = cfg.config_cade["x_dim"]
    y_dim = cfg.config_cade["y_dim"]
    latent_dim = cfg.config_cade["latent_dim"]
    hidden_dim = cfg.config_cade["hidden_dim"]
    p_noise = cfg.config_cade["p_noise"]

    min_noise = cfg.config_cade["min_noise"]
    max_noise = cfg.config_cade["max_noise"]
    noise_schedule_dim = cfg.config_cade["noise_schedule_dim"]
    
    noise_schedule = np.linspace(max_noise, min_noise, num=noise_schedule_dim)
    noise_dim = cfg.config_cade["noise_dim"]
    
    cdae = ConditionalDenoisingAutoencoderV2(x_dim=x_dim, y_dim=y_dim)

    num_epochs = cfg.config_cade["num_epochs"]
    lr = cfg.config_cade["lr"]
    lambda_sparse = cfg.config_cade["lambda_sparse"]

    optimizer = optim.Adam(cdae.parameters(), lr=lr)
    criterion = nn.MSELoss()

    print("parameters: ", count_parameters(cdae))
    print("TRAINING CDAE")

    cdae.train()
    for epoch in range(num_epochs):
        for i, (x, y) in enumerate(dataloader):
            
            optimizer.zero_grad()
            
            if np.random.rand() < p_noise:
                noise_level = torch.tensor(np.random.choice(noise_schedule), dtype=torch.float32).expand(x.size(0), 1)
                y_noisy = y + torch.randn_like(y) * noise_level
            else:
                y_noisy = y
                noise_level = torch.zeros(x.size(0), 1)
                
            
            vec_recon, latent = cdae(x, y_noisy, noise_level)
            
            loss_sparse = torch.mean(torch.abs(latent))
            
            truth = torch.cat([x, y], dim=1)
            
            loss = criterion(truth, vec_recon) + lambda_sparse * loss_sparse
            
            loss.backward()
            optimizer.step()
            
        if (epoch + 1) % 300 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}')
            
            with torch.no_grad():
                val_loss = 0
                for x_val, y_val in val_dataloader:
                    
                    if np.random.rand() < p_noise:
                        noise_level = torch.tensor(np.random.choice(noise_schedule), dtype=torch.float32).expand(x_val.size(0), 1)
                        y_noisy_val = y_val + torch.randn_like(y_val) * noise_level
                    else:
                        y_noisy_val = y_val
                        noise_level = torch.zeros(x_val.size(0), 1)
                    
                    vec_recon_val, _ = cdae(x_val, y_noisy_val, noise_level)
                    
                    truth = torch.cat([x_val, y_val], dim=1)
                    val_loss += criterion(truth, vec_recon_val).item()
                
                val_loss /= len(val_dataloader)
                print(f'Validation Loss CDAE: {val_loss:.6f}')


    cdae.eval()
    loss = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(test_dataloader):
            
            vec_recon, latent = cdae(x, y, torch.zeros(x.size(0), 1))
            
            truth = torch.cat([x, y], dim=1)
            
            loss_value = criterion(truth, vec_recon)
            loss += loss_value.item()
        loss = loss / len(test_dataloader)

    print(f'Test Loss: {loss:.6f}')
    print(f" RMSE: {np.sqrt(loss):.6f}")
    print("Number of parameters: ", count_parameters(cdae))



    ####* Define and train the mapping function

    map_net = MappingNet(input_dim=x_dim, hidden_dim=cfg.config_mapping["hidden_dim"], output_dim=y_dim)

    criterion_map = nn.MSELoss()
    optimizer_map = optim.Adam(map_net.parameters(), lr=lr)

    lambda_reg = cfg.config_mapping["lambda_reg"]
    num_epochs = cfg.config_mapping["num_epochs"]

    map_net.train()
    for epoch in range(num_epochs):
        for i, (x, y) in enumerate(dataloader):
            
            optimizer_map.zero_grad()
            
            y_pred = map_net(x)
            
            loss_reg = torch.norm(map_net.fc1.weight, p=2)**2 + torch.norm(map_net.fc2.weight, p=2)**2
            loss = criterion_map(y_pred, y) + lambda_reg * loss_reg
            
            loss.backward()
            optimizer_map.step()
            
        if (epoch + 1) % 500 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}')
            
            with torch.no_grad():
                val_loss = 0
                for x_val, y_val in val_dataloader:
                    y_pred_val = map_net(x_val)
                    val_loss += criterion_map(y_pred_val, y_val).item()
                
                val_loss /= len(val_dataloader)
                print(f'Validation Loss: {val_loss:.6f}')


    ###* Evaluate the mapping function
    print("EVALUATING MAPPING FUNCTION")

    map_net.eval()

    test_loss_map = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(test_dataloader):
            
            y_pred = map_net(x)
            
            loss_value = criterion_map(y_pred, y)
            test_loss_map += loss_value.item()
        test_loss_map = test_loss_map / len(test_dataloader)

    print(f'Test Loss: {test_loss_map:.6f}')
    print(f"RMSE: {np.sqrt(test_loss_map):.6f}")
    print("Number of parameters: ", count_parameters(map_net))



    # ####* Save the models
    # torch.save(cdae.state_dict(), "cdae_model.pth")
    # torch.save(map_net.state_dict(), "mapping_model.pth")


    ####* Refine the mapping function with the progressive CDAE
    def refine_y_progressive(cdae, x, y_init, noise_schedule, num_iters_per_level=50, step_size=1e-3, eps_conv=1e-3, eps_clip=5e-2):
        y = y_init.clone().detach()
        
        for noise_level in noise_schedule:
            for _ in range(num_iters_per_level):
                
                with torch.no_grad():
                    noise_level_tensor = torch.full((x.size(0), 1), noise_level)
                    vec_recon, _ = cdae(x, y, noise_level_tensor)
                    
                residual_y = vec_recon[:, x_dim:] - y
                
                if torch.norm(residual_y) < eps_conv:
                    break
                
                torch.clip(residual_y, -eps_clip, eps_clip, out=residual_y)
                
                y = y + step_size * residual_y
        return y


    test_refine_loss = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(test_dataloader):
            
            y_pred = map_net(x)

            y_refined = refine_y_progressive(cdae, x, y_pred, noise_schedule, \
                num_iters_per_level=cfg.config_refinement["num_iters_per_level"], step_size=cfg.config_refinement["step_size"], \
                eps_conv=cfg.config_refinement["eps_convergence"], eps_clip=cfg.config_refinement["eps_clip"])
            
            loss_value = criterion_map(y_refined, y)
            test_refine_loss += loss_value.item()
        test_refine_loss = test_refine_loss / len(test_dataloader)

    print(f'Final Test Loss: {test_refine_loss:.6f}')
    print(f"RMSE: {np.sqrt(test_refine_loss):.6f}")
    
    return test_loss_map, test_refine_loss


if __name__ == "__main__":
    
    # seed_vec = np.arange(1, 100, 5)
    seed_vec = np.arange(1, 100, 10)
    
    test_loss_map_vec = []
    test_refine_loss_vec = []
    
    for i, seed in enumerate(seed_vec):
        test_loss_map, test_refine_loss = train_system(seed)
        test_loss_map_vec.append(test_loss_map)
        test_refine_loss_vec.append(test_refine_loss)
        # print("Test Loss Map: ", test_loss_map_vec, "Test Refine Loss: ", test_refine_loss_vec, "Seed: ", seed)
        print("Seed: ", seed, i)
        print("*"*50)
    
    
    mean_test_loss_map = np.mean(np.sqrt(test_loss_map_vec))
    mean_test_refine_loss = np.mean(np.sqrt(test_refine_loss_vec))
    std_test_loss_map = np.std(np.sqrt(test_loss_map_vec))
    std_test_refine_loss = np.std(np.sqrt(test_refine_loss_vec))
    
    with open("cdae_results_multiple_seeds_new.txt", "w") as f:
        f.write("Test Loss Map: " + str(test_loss_map_vec) + "\n")
        f.write("Test Refine Loss: " + str(test_refine_loss_vec) + "\n")
        f.write("Seed: " + str(seed_vec) + "\n")
        
        f.write("RMSQ Mean Test Loss Map: " + str(mean_test_loss_map) + "\n")
        f.write("RMSQ Mean Test Refine Loss: " + str(mean_test_refine_loss) + "\n")
        f.write("RMSQ Std Test Loss Map: " + str(std_test_loss_map) + "\n")
        f.write("RMSQ Std Test Refine Loss: " + str(std_test_refine_loss) + "\n")