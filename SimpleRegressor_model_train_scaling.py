import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import models
import config as cfg

# seed = cfg.config_set["seed"]

# torch.manual_seed(seed)
# np.random.seed(seed)


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



def train_system(seed, hidden_dim):
    
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


    ###* Define the Simple Regressor  and train the model

    x_dim = cfg.config_cade["x_dim"]
    y_dim = cfg.config_cade["y_dim"]



    class SimpleRegressor(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super(SimpleRegressor, self).__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    
    
    model = SimpleRegressor(x_dim, hidden_dim, y_dim)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    n_epochs = 1100
    lambda_reg = 1e-5

    for epoch in range(n_epochs):
        model.train()
        for i, (X, y) in enumerate(dataloader):
            optimizer.zero_grad()
            
            y_pred = model(X)
            loss = criterion(y_pred, y) + lambda_reg * torch.norm(model.fc1.weight, p=2) ** 2 \
                + lambda_reg * torch.norm(model.fc2.weight, p=2) ** 2
            
            loss.backward()
            optimizer.step()
            
            
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for j, (X_val, y_val) in enumerate(val_dataloader):
                y_pred_val = model(X_val)
                val_loss += criterion(y_pred_val, y_val).item()
            
            val_loss /= len(val_dataloader)
            # print(f"Validation Loss: {val_loss}")
            
        # early_stopper(val_loss)
        # if early_stopper.early_stop:
        #     print("Early stopping, epoch: ", epoch)
        #     break
        
    
    test_loss_regressor_direct = 0.0
    model.eval()
    with torch.no_grad():
        for i, (X, y) in enumerate(test_dataloader):
            y_pred = model(X)
            loss = criterion(y_pred, y)
            test_loss_regressor_direct += loss.item()
        
    test_loss_regressor_direct /= len(test_dataloader)
    print(f"Test Loss: {test_loss_regressor_direct}")
    print(f"RMSQ Test Loss: {np.sqrt(test_loss_regressor_direct)}")
    print(f"Number of parameters: {count_parameters(model)}")
    
    return test_loss_regressor_direct, count_parameters(model)


if __name__ == "__main__":
    
    # seed_vec = np.arange(1, 100, 5)
    
    seed_vec = np.arange(1, 100, 20)
    
    hidden_dim_vec = np.arange(1000, 4000, 300)
    print("Hidden dim vector: ", hidden_dim_vec)
    
    test_loss_vec = []
    
    file_name = "simple_regressor_results_scaling_hidden_dim_new.txt"
    
    with open(file_name, "w") as f:
        f.write("Seed: " + str(seed_vec) + "\n")
        f.write("Hidden dim: " + str(hidden_dim_vec) + "\n\n")
        
    
    for i, hidden_dim in enumerate(hidden_dim_vec):
        print("Hidden dim: ", hidden_dim)
        
        test_loss = []
        for j, seed in enumerate(seed_vec):
            loss, params = train_system(seed, hidden_dim)
            test_loss.append(loss)
        
        test_loss_vec.append(test_loss)
        
        with open(file_name, "a") as f:
            f.write("Hidden dim: " + str(hidden_dim) + "\n")
            f.write("Test Loss: " + str(test_loss) + "\n")
            f.write("Number of parameters: " + str(params) + "\n\n")
    
    
    test_loss_vec = np.array(test_loss_vec)
    
    mean_test_loss = np.mean(np.sqrt(test_loss_vec), axis=1)
    std_test_loss = np.std(np.sqrt(test_loss_vec), axis=1)
    
    with open(file_name, "a") as f:
        f.write("Mean Test Loss: " + str(mean_test_loss) + "\n")
        f.write("Std Test Loss: " + str(std_test_loss) + "\n")
    
    