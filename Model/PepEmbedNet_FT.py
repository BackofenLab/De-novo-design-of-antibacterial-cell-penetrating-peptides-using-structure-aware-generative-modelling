import torch
import torch.nn as nn

from Model.model import PeptideEmbedNet
from globals import tokenizer


class PepEmbedNetFT(nn.Module):
    def __init__(self, pretrained_model: PeptideEmbedNet):
        super().__init__()

        self.pretrained = pretrained_model
        # Freeze pre-trained model
        for p in self.pretrained.parameters():
            p.requires_grad = False

        self.out = nn.Linear(self.pretrained.embedding_dim, tokenizer.vocab_size)

    def forward(self, x: torch.tensor):
        with torch.no_grad():
            _, details = self.pretrained(x)
            embeddings = details["embeddings"]

        out = self.out(embeddings)
        return out
