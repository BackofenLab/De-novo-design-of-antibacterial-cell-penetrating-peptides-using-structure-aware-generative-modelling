"""
Script for training TransformerVAE given PeptideEmbedNet and CoordVAE.
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import pandas as pd
import ast
from tqdm import tqdm
import matplotlib.pyplot as plt

from Model.model import PeptideEmbedNet
from Model.vae import TransformerVAE
from Model.TSC_vae import CoordVAE
from transformers import AutoTokenizer, EsmConfig
from globals import device, tokenizer


class SeqDataset(Dataset):
    def __init__(self, codes, pdbs):
        self.codes = codes
        processed = [CoordVAE.preprocess_coords(c) for c in pdbs]
        self.pdbs = torch.tensor(np.array(processed), dtype=torch.float32)

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, idx):
        return self.codes[idx], self.pdbs[idx]


def train_vae(model, train_loader, test_loader, optimizer, pad_token_id, device, epochs=10):
    """
    Trains a TransformerVAE over multiple epochs using a custom loss with reconstruction 
    and KL terms, evaluates on a test set each epoch, logs training progress, tracks 
    performance, and returns a plot of train vs test loss curves.
    """
    train_perf, test_perf = [], []
    kl_train, kl_test = [], []
    recon_train, recon_test = [], [] 
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_kl_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for (x, pdb) in progress_bar:
            x = list(x)
            tokenized_seq = tokenizer(x,
                                    padding="max_length",
                                    max_length=model.pt_model.context_window,
                                    truncation=True,
                                    return_tensors="pt")
            inp_ids = tokenized_seq["input_ids"]

            optimizer.zero_grad()
            logits, mu, logvar = model(tokenized_seq, pdb)
            loss, ce, kl = vae_loss(logits, inp_ids, mu, logvar, pad_token_id)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            epoch_kl_loss += kl.item()
            epoch_recon_loss += ce.item()

            avg_loss = total_loss / (progress_bar.n + 1)
            progress_bar.set_postfix(loss=f"{avg_loss:.4f}")

        train_loss = total_loss / len(train_loader)
        avg_train_recon_loss = epoch_recon_loss / len(train_loader)
        avg_train_kl_loss = epoch_kl_loss / len(train_loader) 

        model.eval()
        test_loss = 0.0
        test_recon_loss = 0.0
        test_kl_loss = 0.0
        with torch.no_grad():
            for (x, pdb) in test_loader:
                x = list(x)
                tokenized_seq = tokenizer(x,
                                          padding="max_length",
                                          max_length=model.pt_model.context_window,
                                          truncation=True,
                                          return_tensors="pt")

                inp_ids = tokenized_seq["input_ids"]

                logits, mu, logvar = model(tokenized_seq, pdb)
                loss, ce, kl = vae_loss(logits, inp_ids, mu, logvar, pad_token_id)
                test_loss += loss.item() 
                test_kl_loss += kl.item()
                test_recon_loss += ce.item()

        test_loss /= len(test_loader)
        test_recon_loss /= len(test_loader)
        test_kl_loss /= len(test_loader)
        print(f"Epoch {epoch+1}: train loss = {train_loss:.4f}, test loss = {test_loss:.4f}, train kl = {avg_train_kl_loss}")

        train_perf.append(train_loss)
        test_perf.append(test_loss)

        kl_train.append(avg_train_kl_loss)
        kl_test.append(test_kl_loss)

        recon_train.append(avg_train_recon_loss)
        recon_test.append(test_recon_loss)

    # Log
    with open("./Checkpoints/log_TransformerVAE.txt", "w") as f:
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

    fig = plt.figure(figsize=(6, 4))
    plt.plot(train_perf, label="Train Loss")
    plt.plot(test_perf, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("VAE Training Performance")
    plt.legend()
    plt.tight_layout()
    fig.savefig("./Checkpoints/vae_loss_curve", dpi=300)
    plt.close(fig)

    # (2) combined recon (left y) + KL (right y), train/test in distinct colors, legend outside
    epochs_arr = np.arange(1, len(kl_train) + 1)
    fig, ax1 = plt.subplots(figsize=(6.5, 4))

    # colors
    rec_train_color = "#2ca02c"   # green
    rec_test_color  = "#98df8a"   # light green
    kl_train_color  = "#ff7f0e"   # orange
    kl_test_color   = "#ffbb78"   # light orange

    # left axis - recon
    ax1.plot(epochs_arr, recon_train, label="Recon Train", color=rec_train_color, linewidth=2)
    ax1.plot(epochs_arr, recon_test,  label="Recon Test",  color=rec_test_color,  linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Reconstruction Loss", color=rec_train_color)
    ax1.tick_params(axis="y", labelcolor=rec_train_color)
    ax1.grid(alpha=0.25, linewidth=0.6)

    # right axis - KL
    ax2 = ax1.twinx()
    ax2.plot(epochs_arr, kl_train, label="KL Train", color=kl_train_color, linewidth=2)
    ax2.plot(epochs_arr, kl_test,  label="KL Test",  color=kl_test_color,  linewidth=2)
    ax2.set_ylabel("KL Divergence", color=kl_train_color)
    ax2.tick_params(axis="y", labelcolor=kl_train_color)

    # merged legend outside
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, ncol=2, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.15))

    plt.title("TransformerVAE: KL vs Reconstruction", pad=10)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig("./Checkpoints/transf_vae_kl_recon_dual_axis.png", dpi=600, bbox_inches="tight")
    plt.close(fig) 


def vae_loss(logits, targets, mu, logvar, pad_token_id):
    """
    Computes the VAE loss as the sum of cross-entropy reconstruction 
    loss and KL divergence, returning the total loss along with its 
    individual components.
    """
    ce_loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1)
    )

    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.size(0)
    return ce_loss + kl_loss, ce_loss, kl_loss


def main(LATENT_DIM, BATCH_SIZE, NUM_HEADS, EPOCHS, LR):
    """
    Sets up and trains a TransformerVAE by loading datasets and 
    pretrained models, building the combined architecture, initializing 
    the optimizer, reporting model details, running the VAE training 
    loop, and returning the trained model along with a performance plot.
    """
    # Configuration
    csv_path = "./Data/3LC/CPP_3LC_FT.csv"
    weight_path = "./Checkpoints/2025_07_03_PeptideEmbedNet_MLM/model_20250703_134624.pth"
    ts_weight_path = "./Checkpoints/2025_07_05_CoordVAE/ts_vae.pt"

    max_len = 100
    train_split_ratio = 0.9

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Dataset and split
    df = pd.read_csv(csv_path, header=0)
    codes = df["Codes"].tolist()
    pdbs = df["pdb_structure"].tolist()

    dataset = SeqDataset(codes, pdbs)
    train_size = int(train_split_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Get Pretrained Models
    pt_model = PeptideEmbedNet(num_layers=1,
                               ffn_dim=512,
                               context_window=64,
                               causal_masking=False)
    pt_model.load_state_dict(torch.load(weight_path))

    ts_model = CoordVAE(latent_dim=LATENT_DIM)
    ts_model.load_state_dict(torch.load(ts_weight_path))


    # Model
    model = TransformerVAE(
        pretrained_embedding_model=pt_model,
        pretrained_structure_model=ts_model,
        latent_dim=LATENT_DIM,
        vocab_size=tokenizer.vocab_size,
        num_heads=NUM_HEADS,
        max_len=max_len
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Info
    print(model)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params:,}")

    # Training
    train_vae(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            pad_token_id=tokenizer.pad_token_id,
            device=device,
            epochs=EPOCHS
          )

    return model


def codes_to_aa(seq: str):
    """
    Converts a space-separated sequence of three-letter amino acid 
    codes into a one-letter sequence and returns it.
    """
    sequence = seq.split(" ")

    map = {
        "A": "ALA",
        "R": "ARG",
        "N": "ASN",
        "D": "ASP",
        "C": "CYS",
        "Q": "GLN",
        "E": "GLU",
        "G": "GLY",
        "H": "HIS",
        "I": "ILE",
        "L": "LEU",
        "K": "LYS",
        "M": "MET",
        "F": "PHE",
        "P": "PRO",
        "S": "SER",
        "T": "THR",
        "W": "TRP",
        "Y": "TYR",
        "V": "VAL",
    }
    map = {v: k for k, v in map.items()}

    new_seq = [map[k] for k in sequence]
    return "".join(new_seq)


def generate_sequence(model, tokenizer, max_len=100, temperature=1.0):
    """
    Generates a protein sequence by decoding from random latent vectors 
    with a TransformerVAE, sampling tokens with temperature scaling, 
    truncating at the [SEP] token, converting to amino acid one-letter 
    codes, and returning the result.
    """
    model.eval()
    sep_token_id = tokenizer.sep_token_id

    with torch.no_grad():
        z1 = torch.randn(1, model.latent_dim).to(device)
        z2 = torch.randn(1, model.latent_dim).to(device)
        logits = model.decode(z1, z2, tgt_len=max_len)
        probs = F.softmax(logits / temperature, dim=-1)

        sampled_ids = torch.multinomial(probs.squeeze(0), num_samples=1).squeeze(1).tolist()

        # Truncate at [SEP]
        if sep_token_id in sampled_ids:
            sep_index = sampled_ids.index(sep_token_id)
            sampled_ids = sampled_ids[:sep_index]

        decoded_seq = tokenizer.decode(sampled_ids, skip_special_tokens=True)

    decoded_seq = codes_to_aa(decoded_seq) 
    return decoded_seq


def produce_dataset(model, tokenizer):
    """
    Generates 1000 protein sequences with the model, logs each attempt, 
    skips invalid ones, collects valid sequences into a dataframe, 
    and saves them to a CSV file for evaluation.
    """
    seqs = []
    for i in range(1000):
        try:
            seq = generate_sequence(model, tokenizer)
            print(seq)

            if len(seq) > 1:
                seqs.append(seq)
        except KeyError:
            print("ERROR")
            continue
    
    seqs = pd.DataFrame(seqs, columns=[["Sequence"]])
    seqs.to_csv("./Data/Eval/VAE_FT_EVAL.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent_dim", type=int, default=32, help="Latent dimension in VAEs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch_size in VAEs")
    parser.add_argument("--num_heads", type=int, default=4, help="Num heads in VAEs")
    parser.add_argument("--epochs", type=int, default=100, help="Epochs in VAEs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="LR in VAEs")

    args = parser.parse_args()

    model = main(LATENT_DIM=args.latent_dim,
                 BATCH_SIZE=args.batch_size,
                 NUM_HEADS=args.num_heads,
                 EPOCHS=args.epochs,
                 LR=args.learning_rate)

    produce_dataset(model, tokenizer) 
    torch.save(model.state_dict(), "./Checkpoints/vae.pth")
