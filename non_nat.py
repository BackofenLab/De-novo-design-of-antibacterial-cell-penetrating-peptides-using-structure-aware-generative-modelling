"""
Script experimenting with PeptideEmbedNet for CPPs with non natural
resiudes. Therefore objective and model where changed from MLM to CLM.
Otherwise similar to main.py.
"""

from datetime import datetime
from typing import Optional, Tuple, List, Union

import sys
import pandas as pd
import random
import matplotlib.pyplot as plt
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
from globals import device
from Model.model import PeptideEmbedNet

from transformers import PreTrainedTokenizer


class CharTokenizer(PreTrainedTokenizer):
    def __init__(
        self,
        csv_path: str,
        seq_column: str = "Sequence",
        pad_token: str = "[PAD]",
        unk_token: str = "[UNK]",
        cls_token: str = "[CLS]",
        sep_token: str = "[SEP]",
        mask_token: str = "[MASK]",
        **kwargs,
    ):
        # Special tokens
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.cls_token = cls_token
        self.sep_token = sep_token
        self.mask_token = mask_token

        self.vocab = {}
        self.ids_to_tokens = {}

        # Build vocab from all characters in CSV column
        self._build_vocab(csv_path, seq_column)

        self.model_input_names = ["input_ids", "attention_mask"]

        super().__init__(
            pad_token=self.pad_token,
            unk_token=self.unk_token,
            cls_token=self.cls_token,
            sep_token=self.sep_token,
            mask_token=self.mask_token,
            **kwargs
        )

    def _build_vocab(self, path: str, seq_column: str):
        df = pd.read_csv(path)
        if seq_column not in df.columns:
            raise ValueError(f"Column '{seq_column}' not found in CSV. Columns: {list(df.columns)}")

        sequences = df[seq_column].dropna().astype(str).tolist()

        chars = set()
        for s in sequences:
            chars.update(list(s))

        # Add special tokens first
        specials = [self.pad_token, self.unk_token, self.cls_token, self.sep_token, self.mask_token]
        for tok in specials:
            self._add_token(tok)

        for ch in sorted(chars):
            if ch not in self.vocab:
                self._add_token(ch)

    def _add_token(self, token: str):
        if token not in self.vocab:
            idx = len(self.vocab)
            self.vocab[token] = idx
            self.ids_to_tokens[idx] = token

    def _tokenize(self, text: str) -> List[str]:
        return list(text)

    def _convert_token_to_id(self, token: str) -> int:
        return self.vocab.get(token, self.vocab[self.unk_token])

    def _convert_id_to_token(self, index: int) -> str:
        return self.ids_to_tokens.get(index, self.unk_token)

    def get_vocab(self):
        return dict(self.vocab)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        return "".join(tokens)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def __call__(
        self,
        text: Union[str, List[str]],
        padding: Union[bool, str] = False,
        max_length: Optional[int] = None,
        truncation: bool = False,
        return_tensors: Optional[str] = None,
    ):
        if isinstance(text, str):
            return self._encode_single(text, max_length, padding, truncation, return_tensors)
        elif isinstance(text, list):
            return self._encode_batch(text, max_length, padding, truncation, return_tensors)
        else:
            raise ValueError("Input must be a string or list of strings")

    def _encode_single(
        self,
        text: str,
        max_length: Optional[int],
        padding: Union[bool, str],
        truncation: bool,
        return_tensors: Optional[str],
    ):
        tokens = [self.cls_token] + self._tokenize(text) + [self.sep_token]
        input_ids = [self._convert_token_to_id(t) for t in tokens]
        attention_mask = [1] * len(input_ids)

        # Truncate
        if max_length is not None and truncation:
            input_ids = input_ids[:max_length]
            attention_mask = attention_mask[:max_length]

        # Pad if needed
        if (padding == "max_length") and (max_length is not None):
            pad_id = self.vocab[self.pad_token]
            if len(input_ids) < max_length:
                pad_len = max_length - len(input_ids)
                input_ids += [pad_id] * pad_len
                attention_mask += [0] * pad_len

        if return_tensors == "pt":
            import torch
            return {
                "input_ids": torch.tensor([input_ids], dtype=torch.long),
                "attention_mask": torch.tensor([attention_mask], dtype=torch.long),
            }
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def _encode_batch(
        self,
        texts: List[str],
        max_length: Optional[int],
        padding: Union[bool, str],
        truncation: bool,
        return_tensors: Optional[str],
    ):
        batch = [self._encode_single(t, max_length, padding, truncation, return_tensors=None) for t in texts]
        input_ids = [ex["input_ids"] for ex in batch]
        attention_mask = [ex["attention_mask"] for ex in batch]

        if return_tensors == "pt":
            import torch
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            }
        return {"input_ids": input_ids, "attention_mask": attention_mask}

tokenizer = CharTokenizer("./Data/CPPSite/validated_cpps_nat_and_nonnat.csv")


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

    train_set, test_set = train_test_split(smiles, test_size=0.2, random_state=42)
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

        # Autoregressive Objective
        targets = input_ids[:, 1:]
        predictions, _ = model(tokenized_seq)
        predictions = predictions[:, :-1, :]

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
    token_to_ignore =  tokenizer.pad_token_id
    criterion = nn.CrossEntropyLoss(ignore_index=token_to_ignore)
    optimizer = optim.AdamW(model.parameters(),
                           lr=1e-4,
                           weight_decay=1e-3)

    num_training_steps = len(train_loader) * epochs
    num_warmup_steps = int(0.1 * num_training_steps)  # 10% warmup
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=num_warmup_steps,
                                                num_training_steps=num_training_steps) 

    train_loss_ot = []
    test_loss_ot = []
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

        train_loss_ot.append(train_loss)
        test_loss_ot.append(test_loss)

        print("Epoch {} - Train Loss: {} - Test Loss: {}".format(epoch, train_loss, test_loss))

        # Checkpoint model after every epoch
        save_model(model)
        log(epoch, len(train_loader), train_loss, test_loss)

    return train_loss_ot, test_loss_ot


def produce_seq(model):
    """
    Autoregressively generates a sequence with the model by sampling 
    tokens step-by-step under temperature scaling until [SEP] or maximum 
    length, printing tokens as they are produced and returning the final 
    decoded string.
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
            break

        pred += next_token_str
    
    pred = pred[:-1]
    return pred


def aa_main():
    """
    Trains a PeptideEmbedNet on validated CPP sequences (natural and non-natural) 
    using a causal language modeling setup, reports trainable parameters, and generates 
    example sequences after training.
    """
    train_loader, _, test_loader = gather_data("./Data/CPPSite/validated_cpps_nat_and_nonnat.csv", True, datapoints=None, col_name="Sequence")

    pepEmbedNet = PeptideEmbedNet(
            vocab_size=tokenizer.vocab_size,
            num_layers=1,
            ffn_dim=512,
            context_window=64,
            causal_masking=True,
    ).to(device)

    print(pepEmbedNet)
    print("{} trainable parameters".format(sum(p.numel() for p in pepEmbedNet.parameters() if
                                               p.requires_grad)))
    print("")

    train_loss_ot, test_loss_ot = train(50, pepEmbedNet, train_loader, test_loader)

    plt.figure(figsize=(6, 4))
    plt.plot(train_loss_ot, label="Train Loss")
    plt.plot(test_loss_ot, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train and Test Loss over Epochs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("./Checkpoints/loss_over_time.png", dpi=300) 

    create_examples(pepEmbedNet)


def create_examples(model):
    """
    Generates 1000 sequences with the model, stores them in a dataframe under 
    the column "Sequence", and saves the results to a CSV file for evaluation.
    """
    seqs = []
    natural_counterparts = []
    
    non_nat_residues = ["O", "r", "h"]

    counterpart_dict = {
            "O": "K",
            "r": "R",
            "h": "H",
            }
    

    while len(seqs) != 1000:
        seq = produce_seq(model)
        if any(c in seq for c in non_nat_residues) and all(c not in seq for c in ["[", "]", "(", ")"]):
            if seq not in [s[0] for s in seqs]:
                nat_seq = ''.join(counterpart_dict.get(c, c) for c in seq)
                seqs.append([seq, nat_seq])
                print(seq + " - " + nat_seq)
            
    seqs = pd.DataFrame(seqs, columns=["Sequence", "Nat_Counterpart"])
    seqs.to_csv("./Data/Eval/NON_NAT.csv", index=False)


if __name__ == "__main__":
    aa_main()

