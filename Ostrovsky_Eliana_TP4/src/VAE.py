import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from preprocessing import stratified_split

# ----------- Modelos -----------

class DeepVAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=512, latent_dim=2):
        super(DeepVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

class DeepVAE_Regularized(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=512, latent_dim=2):
        super(DeepVAE_Regularized, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# ----------- Funciones auxiliares -----------

def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + beta * KLD

def train_and_get_latents(model_class, X_train, X_test, y_test, device, beta=1.0, epochs=50, label="VAE"):
    model = model_class().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train)), batch_size=128, shuffle=True)
    test_loader = DataLoader(TensorDataset(torch.tensor(X_test)), batch_size=128)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            x = batch[0].to(device)
            optimizer.zero_grad()
            recon_x, mu, logvar = model(x)
            loss = vae_loss(recon_x, x, mu, logvar, beta)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[{label}] Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_loader.dataset):.4f}")

    # Inferencia
    model.eval()
    mus = []
    with torch.no_grad():
        for batch in test_loader:
            x = batch[0].to(device)
            mu, _ = model.encode(x)
            mus.append(mu.cpu().numpy())
    return np.concatenate(mus)

def plot_latent_comparison(mus_list, y_test, titles, save_path=None):
    fig, axes = plt.subplots(1, len(mus_list), figsize=(6 * len(mus_list), 5))
    if len(mus_list) == 1:
        axes = [axes]
    for ax, mus, title in zip(axes, mus_list, titles):
        for digit in np.unique(y_test):
            idx = y_test == digit
            ax.scatter(mus[idx, 0], mus[idx, 1], s=10, alpha=0.5, label=str(digit))
        ax.set_title(title)
        ax.set_xlabel("mu[0]")
        ax.set_ylabel("mu[1]")
        ax.grid(True)
    axes[0].legend(title="Etiqueta")
    plt.suptitle("Comparación del espacio latente - distintas variantes de VAE")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

# ----------- Ejecución -----------

# Datos
df = pd.read_csv("Ostrovsky_Eliana_TP4/data/MNIST_dataset.csv" \
"")
X = df.drop(columns=["label"]).values.astype(np.float32) / 255.0
y = df["label"].values
X_train, X_test, y_train, y_test = stratified_split(X, y, train_ratio=0.8, test_ratio=0.2, random_state=42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Entrenar y recolectar latentes
mus_base = train_and_get_latents(DeepVAE, X_train, X_test, y_test, device, beta=1.0, epochs=20, label="Base")
mus_beta = train_and_get_latents(DeepVAE, X_train, X_test, y_test, device, beta=4.0, epochs=50, label="β-VAE")
mus_bn_do = train_and_get_latents(DeepVAE_Regularized, X_train, X_test, y_test, device, beta=1.0, epochs=50, label="BatchNorm+Dropout")

# Visualizar
plot_latent_comparison(
    [mus_base, mus_beta, mus_bn_do],
    y_test,
    ["VAE Profundo (β=1.0, 20 ep.)", "β-VAE (β=4.0, 50 ep.)", "VAE Prof. con BN+Dropout (β=1.0, 50 ep.)"],
    save_path="vae_latent_comparison.png"
)
