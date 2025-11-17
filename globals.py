"""
Script for capturing global variables shared across models and
introduces the tokenizer for PeptideEmbedNet.
"""

from typing import List
import pandas as pd
import torch
from transformers import AutoTokenizer, PreTrainedTokenizer


BATCH_SIZE = 32
# DATA_PATH = r"./Data/example_data.csv"
DATA_PATH = r"./Data/3LC/3LC.csv"
MODEL_PATH = r"Checkpoints/"

"""
model_checkpoint = "DeepChem/ChemBERTa-77M-MTR"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
"""

class CodeTokenizer(PreTrainedTokenizer):
    def __init__(self, data_path: str = DATA_PATH, **kwargs):
        # Special tokens
        self.pad_token = "[PAD]"
        self.unk_token = "[UNK]"
        self.cls_token = "[CLS]"
        self.sep_token = "[SEP]"
        self.mask_token = "[MASK]"

        self.vocab = {}
        self.ids_to_tokens = {}
        self._build_vocab(data_path)
        self.model_input_names = ["input_ids", "attention_mask"]

        super().__init__(
            pad_token=self.pad_token,
            unk_token=self.unk_token,
            cls_token=self.cls_token,
            sep_token=self.sep_token,
            mask_token=self.mask_token,
            **kwargs
        )

    def _build_vocab(self, path):
        df = pd.read_csv(path)
        tokens = set()
        for seq in df["Codes"]:
            tokens.update(seq.split("#"))
        tokens = sorted(tokens)

        # Build vocab
        special_tokens = [self.pad_token, self.unk_token, self.cls_token, self.sep_token, self.mask_token]
        for token in special_tokens + tokens:
            self._add_token(token)

    def _add_token(self, token):
        if token not in self.vocab:
            idx = len(self.vocab)
            self.vocab[token] = idx
            self.ids_to_tokens[idx] = token

    def _tokenize(self, text: list | str) -> List[str]:
        return text.split("#")

    def _convert_token_to_id(self, token: str) -> int:
        return self.vocab.get(token, self.vocab[self.unk_token])

    def _convert_id_to_token(self, index: int) -> str:
        return self.ids_to_tokens.get(index, self.unk_token)

    def get_vocab(self):
        return self.vocab

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        return " ".join(tokens)

    def __call__(self, text, max_length=None, padding=True, truncation=True, return_tensors=None):
        if isinstance(text, str):
            return self._encode_single(text, max_length, padding, truncation, return_tensors)
        elif isinstance(text, list):
            return self._encode_batch(text, max_length, padding, truncation, return_tensors)
        else:
            raise ValueError("Input must be a string or list of strings")

    def _encode_single(self, text, max_length, padding, truncation, return_tensors):
        tokens = [self.cls_token] + self._tokenize(text) + [self.sep_token]
        input_ids = [self._convert_token_to_id(t) for t in tokens]
        attention_mask = [1] * len(input_ids)

        if max_length:
            if truncation:
                input_ids = input_ids[:max_length]
                attention_mask = attention_mask[:max_length]
            if padding == "max_length":
                pad_len = max_length - len(input_ids)
                input_ids += [self.vocab[self.pad_token]] * pad_len
                attention_mask += [0] * pad_len

        if return_tensors == "pt":
            import torch
            return {
                "input_ids": torch.tensor([input_ids]),
                "attention_mask": torch.tensor([attention_mask]),
            }
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    def _encode_batch(self, texts, max_length, padding, truncation, return_tensors):
        batch = [self._encode_single(t, max_length, padding, truncation, return_tensors=None) for t in texts]
        input_ids = [ex["input_ids"] for ex in batch]
        attention_mask = [ex["attention_mask"] for ex in batch]

        if return_tensors == "pt":
            import torch
            return {
                "input_ids": torch.tensor(input_ids),
                "attention_mask": torch.tensor(attention_mask),
            }
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    @property
    def vocab_size(self):
        return len(self.vocab)

tokenizer = CodeTokenizer()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
