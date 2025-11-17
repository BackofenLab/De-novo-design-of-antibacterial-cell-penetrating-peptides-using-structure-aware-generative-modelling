"""
Legacy script for training PeptideEmbedNet.
"""

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

import globals
from globals import device, tokenizer
from Model.model import PeptideEmbedNet
from Model.PepEmbedNet_FT import PepEmbedNetFT


def map_code_to_aa(seq):
    """
    Converts a sequence of three-letter amino acid codes (separated by #) 
    into a one-letter sequence, returning an empty list if an unknown code 
    is encountered.
    """
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
    inverted = {v: k for k, v in map.items()}

    seq = seq.split("#")
    new_seq = []
    for code in seq:
        try:
            new_seq.append(inverted[code])
        except KeyError:
            return []
    return "".join(new_seq)


def produce_seq(model):
    """
    Generates a protein sequence autoregressively with a transformer model 
    by sampling tokens step-by-step until a stop token is reached, converts 
    the output into one-letter amino acid codes, and returns the final sequence.
    """
    max_len = 128
    temperature = 1.0

    start_seq = ""
    tokenized = tokenizer(start_seq,
                          return_tensors="pt",
                          padding="max_length",
                          truncation=True,
                          max_length=model.pretrained.context_window)
    input_ids = tokenized["input_ids"][:, :1].to(device)
    attention_mask = tokenized["attention_mask"][:, :1].to(device)

    pred = ""
    for i in range(max_len):
        out = model({"input_ids": input_ids,
                     "attention_mask": attention_mask})
        logits = out[:, -1, :] / temperature

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).to(device)
        next_token_str = tokenizer.decode(next_token[0])

        input_ids = torch.cat((input_ids, next_token), dim=1).to(device)
        next_mask = torch.ones((attention_mask.size(0), 1), dtype=attention_mask.dtype).to(device)
        attention_mask = torch.cat((attention_mask, next_mask), dim=1)

        if next_token_str == "[SEP]":
            print("")
            break
        
        pred += next_token_str + "#"
    
    pred = pred[:-1]
    pred = map_code_to_aa(pred)
    print(pred)
    return pred


def train(model: PepEmbedNetFT,
          train_loader: DataLoader,
          test_loader: DataLoader,
          num_epochs: int = 5):
    """
    Trains and evaluates the PepEmbedNetFT model for a given number 
    of epochs using cross-entropy loss with padding ignored, the AdamW 
    optimizer, and tokenized input batches, reporting average training 
    and test losses per epoch.
    """

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_train_loss = 0
        train_bar = tqdm(train_loader, leave=False, desc=f"Epoch {epoch}/{num_epochs} [Training]")

        for x_batch in train_bar:
            tokenized_seq = tokenizer(x_batch,
                                      padding="max_length",
                                      max_length=model.pretrained.context_window,
                                      truncation=True,
                                      return_tensors="pt")
            input_ids = tokenized_seq["input_ids"].to(device)
            targets = input_ids[:, 1:]
            preds = model(tokenized_seq)
            preds = preds[:, :-1, :]

            loss = criterion(preds.reshape(-1, preds.size(-1)), targets.reshape(-1))
            total_train_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = total_train_loss / len(train_loader)

        # Evaluation
        model.eval()
        total_test_loss = 0
        with torch.no_grad():
            for x_batch in test_loader:
                tokenized_seq = tokenizer(x_batch,
                                          padding="max_length",
                                          max_length=model.pretrained.context_window,
                                          truncation=True,
                                          return_tensors="pt")
                input_ids = tokenized_seq["input_ids"].to(device)
                targets = input_ids[:, 1:]
                preds = model(tokenized_seq)
                preds = preds[:, :-1, :]

                loss = criterion(preds.reshape(-1, preds.size(-1)), targets.reshape(-1))
                total_test_loss += loss.item()

        avg_test_loss = total_test_loss / len(test_loader)
        print(f"Epoch {epoch}/{num_epochs} — Train Loss: {avg_train_loss:.4f} — Test Loss: {avg_test_loss:.4f}")


def get_pretrained_model():
    """
    Loads a pretrained PeptideEmbedNet model with specified architecture 
    parameters, restores its weights from a checkpoint, sets it to 
    evaluation mode, and returns it.
    """
    pepEmbedNet = PeptideEmbedNet(
            num_layers=1,
            ffn_dim=512,
            context_window=64,
    ).to(device)

    pepEmbedNet.load_state_dict(torch.load("./Checkpoints/2025_08_06_AA/AA_P1602586_T250k.pth"))
    pepEmbedNet.eval()
    return pepEmbedNet


def create_examples(model):
    """
    Generates 1000 protein sequences with the model, filters out 
    invalid or too-short outputs, stores valid sequences in a dataframe, 
    and saves them to a CSV file for evaluation.
    """
    seqs = []
    for i in range(1000):
        seq = produce_seq(model)
        if seq != [] and len(seq) > 1:
            seqs.append(seq)
    seqs = pd.DataFrame(seqs, columns=["Sequence"])
    seqs.to_csv("./Data/PepEmbedNet_FT_Eval.csv", index=False)


if __name__ == "__main__":
    FT_DATASET = "./Data/CPP_3LC_FT.csv"
    data = pd.read_csv(FT_DATASET)["Codes"].tolist()
    train_set, test_set = train_test_split(data, test_size=0.1, random_state=42)
    train_loader = DataLoader(train_set, batch_size=globals.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=globals.BATCH_SIZE, shuffle=True)

    pt_model = get_pretrained_model() 
    ft_model = PepEmbedNetFT(pt_model)

    print(ft_model)
    print("{} trainable parameters".format(sum(p.numel() for p in ft_model.parameters() if
                                               p.requires_grad)))
    print("")

    train(ft_model, train_loader, test_loader, 100)

    ft_model.eval()
    create_examples(ft_model) 



