"""
Up to date script for training PeptideEmbedNet. Makes use of MLM.
"""

from datetime import datetime
from typing import Optional, Tuple

import sys
import pandas as pd
import random
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules.loss import _WeightedLoss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, StepLR
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from rdkit import Chem

import globals
from globals import device, tokenizer
from Model.model import PeptideEmbedNet


def produce_skiprows(data_path: str, n: int):
    """
    Function for generating skiprows, used to save memory during testing
    
    :param data_path: The path to the csv
    :param n: Num of datapoints to keep
    """

    with open(data_path) as f:
        total_lines = sum(1 for _ in f) - 1

    skip = sorted(random.sample(range(1, total_lines+1), total_lines - n))
    return skip



def gather_data(data_path: str, debug: bool = False, datapoints: int=None, col_name:str = "Smiles") -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Gathers data and prepares it to be used in the model

    :param data_path: Path to the csv containing the data
    :param debug: En/Disables logging to console
    :param datapoints: Num of DP to keep from dataset
    :return: train, val and test dataloaders
    """
    if datapoints:
        skiprows = produce_skiprows(data_path, datapoints)
    else:
        skiprows = None
    smiles = pd.read_csv(data_path, index_col=None, skiprows=skiprows)[col_name].tolist()

    train_set, test_set = train_test_split(smiles, test_size=0.01, random_state=42)
    train_set, val_set = train_test_split(train_set, test_size=0.01, random_state=42)

    train_loader = DataLoader(train_set, batch_size=globals.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=globals.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=globals.BATCH_SIZE, shuffle=False)

    if debug:
        print("Datasets:")
        print("Original Data: {}".format(len(smiles)))
        print("Train Set: {} entries".format(len(train_set)))
        print("Validation Set: {} entries".format(len(val_set)))
        print("Test Set: {} entries".format(len(test_set)))
        print("")

    return train_loader, val_loader, test_loader


def save_model(model: nn.Module):
    """
    Saves model to globals.MODEL_PATH. Adds current time stamp to name

    :param model: The model to save
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    torch.save(model.state_dict(), globals.MODEL_PATH + f"model_{timestamp}.pth")


def log(epoch: int, step: int, avg_loss: float, avg_test_loss: Optional[float] = torch.nan):
    """
    Logs relevant info to log.txt in globals.MODEL_PATH

    :param epoch: Current Epoch
    :param step: Current Step
    :param avg_loss: Average Train Loss
    :param avg_test_loss: Average Test Loss
    """
    with open(globals.MODEL_PATH + f"log.txt", "a", encoding="utf-8") as f:
        f.write("Epoch: {}, Step: {}, Train Loss: {}, Test Loss: {}\n".format(epoch,
                                                                              step,
                                                                              avg_loss,
                                                                              avg_test_loss))


def mask_tokens(inputs, mlm_probability=0.15):
    labels = inputs.clone()

    # Create a mask of tokens to predict
    probability_matrix = torch.full(labels.shape, mlm_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
        for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # Only compute loss on masked tokens

    # Replace 80% with [MASK], 10% random, 10% original
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.mask_token_id

    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    return inputs, labels



def run_epoch(model: nn.Module,
              dataloader: DataLoader,
              criterion: _WeightedLoss,
              optimizer: Optional[Optimizer] = None,
              scheduler: Optional[LRScheduler] = None,
              train: Optional[bool] = False,
              desc: Optional[str] = "",
              save_every: Optional[int] = 1000,
              epoch: Optional[int] = None):
    """
    Handles logic for running an epoch with pytorch.

    :param model: The Model to run the epoch on
    :param dataloader: The dataset to run the epoch on
    :param criterion: Loss function
    :param optimizer: Optimizer to be used during training
    :param scheduler: Scheduler for setting learning rate during training
    :param train: Bool determining if backpropagation is conducted
    :param desc: The description for the tqdm progress bar
    :param save_every: Checkpoint model every x steps
    :param epoch: The current epoch - needed for logging
    :return: average loss during the epoch
    """
    if train:
        if optimizer is None:
            raise ValueError("Optimizer must be set in run_epoch if train is True")
    else:
        model.eval()

    epoch_loss = 0
    progress_bar = tqdm(dataloader, leave=False, desc=desc)
    current_lr = None

    for step, x_batch in enumerate(progress_bar):
        tokenized_seq = tokenizer(x_batch,
                                  padding="max_length",
                                  max_length=model.context_window,
                                  truncation=True,
                                  return_tensors="pt")

        input_ids = tokenized_seq["input_ids"].to(device)

        """
        # Autoregressive Objective
        targets = input_ids[:, 1:]
        predictions, _ = model(tokenized_seq)
        predictions = predictions[:, :-1, :]
        """

        # MLM Objective
        mlm_inputs, labels = mask_tokens(input_ids)
        mlm_inputs = mlm_inputs.to(device)
        targets = labels.to(device)
        tokenized_seq["input_ids"] = mlm_inputs
        predictions, _  = model(tokenized_seq)


        loss = criterion(predictions.reshape(-1, predictions.size(-1)), targets.reshape(-1))

        if train:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if scheduler:
                scheduler.step()

            current_lr = optimizer.param_groups[0]['lr']

            # SAVE MODEL CHECKPOINT AND LOGGING
            if step % save_every == 0:
                save_model(model)
                if step != 0:
                    avg_loss = epoch_loss / step
                    log(epoch, step, avg_loss)

        epoch_loss += loss.item()
        if current_lr:
            progress_bar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{current_lr:.6f}")
        else:
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    model.train()
    return epoch_loss / len(dataloader)


def train(epochs: int,
          model: nn.Module,
          train_loader: DataLoader,
          test_loader: DataLoader):
    """
    Main function coordinating the training of the PeptideEmbedNet model.

    :param epochs: Number of epochs to train for
    :param model: The model to train
    :param train_loader: Training data
    :param test_loader: test data
    """
    token_to_ignore = -100  # tokenizer.pad_token_id for Autoregressive
    criterion = nn.CrossEntropyLoss(ignore_index=token_to_ignore)
    optimizer = optim.AdamW(model.parameters(),
                           lr=1e-4,
                           weight_decay=1e-3)

    num_training_steps = len(train_loader) * epochs
    num_warmup_steps = int(0.1 * num_training_steps)  # 10% warmup
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=num_warmup_steps,
                                                num_training_steps=num_training_steps) 

    for epoch in range(epochs):
        if epoch == 0:
            test_loss = run_epoch(model=model,
                                  dataloader=test_loader,
                                  criterion=criterion,
                                  train=False,
                                  desc="Test")
            log(epoch, 0, torch.nan, test_loss)

        train_loss = run_epoch(model=model,
                               dataloader=train_loader,
                               criterion=criterion,
                               optimizer=optimizer,
                               scheduler=scheduler,
                               train=True,
                               desc="Train",
                               epoch=epoch)

        test_loss = run_epoch(model=model,
                              dataloader=test_loader,
                              criterion=criterion,
                              train=False,
                              desc="Test")

        print("Epoch {} - Train Loss: {} - Test Loss: {}".format(epoch, train_loss, test_loss))

        # Checkpoint model after every epoch
        save_model(model)
        log(epoch, len(train_loader), train_loss, test_loss)


def produce_seq(model):
    """
    Autoregressively generates a sequence token by token using the 
    model, sampling from the output distribution with temperature 
    scaling until a [SEP] token or maximum length is reached, and 
    returns the decoded sequence as a #-separated string.
    """
    max_len = 128
    temperature = 1.0

    start_seq = ""
    tokenized = tokenizer(start_seq,
                          return_tensors="pt",
                          padding="max_length",
                          truncation=True,
                          max_length=model.context_window)
    input_ids = tokenized["input_ids"][:, :1].to(device)
    attention_mask = tokenized["attention_mask"][:, :1].to(device)

    pred = ""
    for i in range(max_len):
        out, _ = model({"input_ids": input_ids,
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

        if i == max_len:
            print("")
            print("Forced end")
        
        print(next_token_str, end=" ")
        pred += next_token_str + "#"
    
    pred = pred[:-1]
    return pred


def smi_main():
    """
    Initializes a PeptideEmbedNet model, reports its trainable parameters, 
    trains it on sequence data using predefined loaders, and generates 
    example sequences after training. !Needs smiles data!
    """
    train_loader, _, test_loader = gather_data(globals.DATA_PATH, True)

    # num_layers = num of transformer layers,
    # ffn_dim = transformer internal embedding dim
    pepEmbedNet = PeptideEmbedNet(
        num_layers=1,
        ffn_dim=1024
    ).to(device)
    print(pepEmbedNet)
    print("{} trainable parameters".format(sum(p.numel() for p in pepEmbedNet.parameters() if
                                               p.requires_grad)))
    print("")

    train(3, pepEmbedNet, train_loader, test_loader)
    create_examples(pepEmbedNet)


def aa_main():
    """
    Trains a PeptideEmbedNet on 3-letter amino acid code data by preparing 
    loaders from the dataset, initializing the model with specified 
    parameters, reporting trainable parameters, and running the 
    training loop.
    """
    train_loader, _, test_loader = gather_data(globals.DATA_PATH, True, datapoints=int(2.5e5), col_name="Codes")

    pepEmbedNet = PeptideEmbedNet(
            num_layers=1,
            ffn_dim=512,
            context_window=64,
       
            causal_masking=False, # True for Autoregressive, False for MLM
    ).to(device)

    print(pepEmbedNet)
    print("{} trainable parameters".format(sum(p.numel() for p in pepEmbedNet.parameters() if
                                               p.requires_grad)))
    print("")

    train(3, pepEmbedNet, train_loader, test_loader)


def create_examples(model):
    """
    Generates 1000 sequences with the model, stores them in a 
    dataframe under the column "Codes", and saves the results 
    to a CSV file.
    """
    seqs = []
    for i in range(1000):
        seqs.append(produce_seq(model))
    seqs = pd.DataFrame(seqs, columns=["Codes"])
    seqs.to_csv("./Data/TESTING.csv", index=False)


if __name__ == "__main__":
    aa_main()






