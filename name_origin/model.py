import torch
from torch import nn
from .common import MAX_LENGTH


def build_vocab(names):
    return {char: index + 2 for index, char in enumerate(sorted(set(''.join(names))))}


def encode(names, vocab):
    x = torch.zeros(len(names), MAX_LENGTH, dtype=torch.long)
    lengths = torch.tensor([len(name) for name in names], dtype=torch.long)
    for i, name in enumerate(names):
        if not 1 <= len(name) <= MAX_LENGTH:
            raise ValueError('Invalid normalized name length')
        x[i, :len(name)] = torch.tensor([vocab.get(char, 1) for char in name])
    return x, lengths


class NameLSTM(nn.Module):
    def __init__(self, vocab_size, num_classes, embedding_dim=32, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(.3)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, lengths):
        output, _ = self.lstm(self.embedding(x))
        # Select the last real character, not the final padding position.
        last = output[torch.arange(len(x), device=x.device), lengths - 1]
        return self.classifier(self.dropout(last))
