import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from io import StringIO
from Bio.PDB import PDBParser

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split


class CoordVAE(nn.Module):
    def __init__(self, latent_dim=1):
        super(CoordVAE, self).__init__()
        self.input_len = 100
        self.input_dim = self.input_len * 3  # Flattened (N x 3)
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 512),
            nn.ReLU(),
            nn.Linear(512, self.input_dim)
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar
    
    @staticmethod
    def extract_ca_coords(pdb_str):
        parser = PDBParser(QUIET=True)
        handle = StringIO(pdb_str)
        structure = parser.get_structure("model", handle)
        coords = []

        for atom in structure.get_atoms():
            if atom.get_id() == "CA":
                coords.append(atom.get_coord())

        return np.array(coords)

    @staticmethod
    def preprocess_coords(pdb_str, max_len=100):
        coords = CoordVAE.extract_ca_coords(pdb_str)
        coords = coords[:max_len]
        if coords.shape[0] < max_len:
            pad = np.zeros((max_len - coords.shape[0], 3))
            coords = np.vstack([coords, pad])
        return coords.flatten()


def vae_loss(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x, reduction="mean")
    kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_div, recon_loss, kl_div


def train(coords, LATENT_DIM, BATCH_SIZE, EPOCHS, LR):
    # Split data
    X_train, X_test = train_test_split(coords, test_size=0.2, random_state=42)
    train_tensor = torch.tensor(np.array(X_train), dtype=torch.float32)
    test_tensor = torch.tensor(np.array(X_test), dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(train_tensor), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_tensor), batch_size=BATCH_SIZE)

    # Setup
    model = CoordVAE(latent_dim=LATENT_DIM)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Train loop
    train_perf, test_perf = [], []
    kl_train, kl_test = [], []
    recon_train, recon_test = [], []
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_kl_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            x = batch[0].to(device)
            optimizer.zero_grad()
            recon_x, mu, logvar = model(x)
            loss, recon_loss, kl_loss = vae_loss(recon_x, x, mu, logvar)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_kl_loss += kl_loss.item()
            epoch_recon_loss += recon_loss.item()
        avg_train_loss = epoch_loss / len(train_loader)
        avg_train_recon_loss = epoch_recon_loss / len(train_loader)
        avg_train_kl_loss = epoch_kl_loss / len(train_loader)

        # Test after each epoch
        model.eval()
        test_loss = 0.0
        test_recon_loss = 0.0
        test_kl_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                x = batch[0].to(device)
                recon_x, mu, logvar = model(x)
                loss, recon_loss, kl_loss = vae_loss(recon_x, x, mu, logvar)
                test_loss += loss.item()
                test_recon_loss += recon_loss.item()
                test_kl_loss += kl_loss.item()

        avg_test_loss = test_loss / len(test_loader)
        avg_test_recon_loss = test_recon_loss / len(test_loader)
        avg_test_kl_loss = test_kl_loss / len(test_loader)

        train_perf.append(avg_train_loss)
        test_perf.append(avg_test_loss)
        
        kl_train.append(avg_train_kl_loss)
        kl_test.append(avg_test_kl_loss)

        recon_train.append(avg_train_recon_loss)
        recon_test.append(avg_test_recon_loss)

        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}")


    out_dir = BASE_DIR / "../Checkpoints" 

    # Log
    with open(out_dir / "log_CoordVAE.txt", "w") as f:
        for epoch, (train_loss, test_loss,
                kl_loss_train, kl_loss_test,
                recon_loss_train, recon_loss_test) in enumerate(zip(train_perf, test_perf,
                                                                    kl_train, kl_test,
                                                                    recon_train, recon_test), 1):
                    f.write(f"Epoch: {epoch}\t"
                            f"Train Loss: {train_loss:.4f}\t"
                            f"Test Loss: {test_loss:.4f}\t"
                            f"KL Loss Train: {kl_loss_train:.4f}\t"
                            f"KL Loss Test: {kl_loss_test:.4f}\t"
                            f"Recon Loss Train: {recon_loss_train:.4f}\t"
                            f"Recon Loss Test: {recon_loss_test:.4f}\n")  

    # (1) Total VAE loss
    fig = plt.figure(figsize=(6, 4))
    plt.plot(train_perf, label="Train Loss")
    plt.plot(test_perf, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Tertiary Structure VAE: Total Loss")
    plt.legend(frameon=False)
    plt.tight_layout()
    fig.savefig(out_dir / "ts_vae_total_loss.png", dpi=300)


    # (2) Recon and KL losses
    epochs = np.arange(1, len(kl_train) + 1)
    fig, ax1 = plt.subplots(figsize=(6.5, 4))

    # --- Style setup ---
    plt.rcParams.update({
        "axes.edgecolor": "#444444",
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.labelcolor": "#222222",
        "xtick.color": "#222222",
        "ytick.color": "#222222",
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 300,
        "font.family": "Helvetica"
    })

    # --- Colors ---
    rec_train_color = "#2ca02c"  # rich green
    rec_test_color  = "#98df8a"  # lighter green
    kl_train_color  = "#ff7f0e"  # strong orange
    kl_test_color   = "#ffbb78"  # lighter orange

    # --- Left y-axis: Reconstruction loss ---
    ax1.plot(epochs, recon_train, label="Recon Train", color=rec_train_color, linewidth=2)
    ax1.plot(epochs, recon_test,  label="Recon Test",  color=rec_test_color, linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Reconstruction Loss", color=rec_train_color)
    ax1.tick_params(axis="y", labelcolor=rec_train_color)
    ax1.grid(alpha=0.25, linewidth=0.6)

    # --- Right y-axis: KL divergence ---
    ax2 = ax1.twinx()
    ax2.plot(epochs, kl_train, label="KL Train", color=kl_train_color, linewidth=2)
    ax2.plot(epochs, kl_test,  label="KL Test",  color=kl_test_color, linewidth=2)
    ax2.set_ylabel("KL Divergence", color=kl_train_color)
    ax2.tick_params(axis="y", labelcolor=kl_train_color)

    # --- Merge legend handles from both axes ---
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    all_lines  = lines_1 + lines_2
    all_labels = labels_1 + labels_2

    # --- Place legend outside plot ---
    ax1.legend(
        all_lines, all_labels,
        frameon=False,
        ncol=2,
        fontsize=9,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15)
    )

    # --- Title and layout ---
    plt.title("CoordVAE: KL vs Reconstruction", fontsize=11, pad=10)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(out_dir / "ts_vae_kl_recon_dual_axis.png", dpi=600, bbox_inches="tight")
    plt.close(fig)

    return model


def coords_to_figure(coords, max_len=100):
    coords = coords[:max_len]
    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], marker='o', markersize=2)
    ax.axis('off')
    return fig


def plot_real_vs_reconstructed(model, coords, n_examples=3, max_len=100):
    model.eval()
    device = next(model.parameters()).device

    indices = np.random.choice(len(coords), size=n_examples, replace=False)
    inputs = torch.tensor(np.array(coords)[indices], dtype=torch.float32).to(device)

    with torch.no_grad():
        reconstructions, _, _ = model(inputs)

    fig = plt.figure(figsize=(6, 2 * n_examples))

    for i in range(n_examples):
        real = inputs[i].cpu().numpy().reshape(-1, 3)[:max_len]
        recon = reconstructions[i].cpu().numpy().reshape(-1, 3)[:max_len]

        real = real[~np.all(real == 0, axis=1)]
        recon = recon[~np.all(recon == 0, axis=1)]

        # Original
        ax_real = fig.add_subplot(n_examples, 2, 2 * i + 1, projection='3d')
        ax_real.plot(real[:, 0], real[:, 1], real[:, 2], marker='o', markersize=2)
        ax_real.set_title(f"Real #{i+1}")
        ax_real.axis('off')

        # Reconstruction
        ax_recon = fig.add_subplot(n_examples, 2, 2 * i + 2, projection='3d')
        ax_recon.plot(recon[:, 0], recon[:, 1], recon[:, 2], marker='o', markersize=2)
        ax_recon.set_title(f"Reconstructed #{i+1}")
        ax_recon.axis('off')

    plt.tight_layout()
    return fig 



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent_dim", type=int, default=16, help="Latent dim for TSC-VAE")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for TSC-VAE")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="LR for TSC-VAE")
    parser.add_argument("--epochs", type=int, default=500, help="Epochs for TSC-VAE")
    args = parser.parse_args()

    BASE_DIR = Path(__file__).resolve().parent
    DATA_PATH = BASE_DIR / "../Data/CPP_AAS_with_structures.csv"

    df = pd.read_csv(DATA_PATH, header=0)
    pdb_strings = df["pdb_structure"].tolist()
    coords = [CoordVAE.preprocess_coords(c) for c in pdb_strings]

    model = train(coords,
                  LATENT_DIM=args.latent_dim,
                  BATCH_SIZE=args.batch_size,
                  LR=args.learning_rate,
                  EPOCHS=args.epochs)

    example_fig = plot_real_vs_reconstructed(model, coords, n_examples=3)
    torch.save(model.state_dict(), BASE_DIR / "../Checkpoints/ts_vae.pt")
    example_fig.savefig(BASE_DIR / "../Checkpoints/ts_vae_examples.png", dpi=300)

    

