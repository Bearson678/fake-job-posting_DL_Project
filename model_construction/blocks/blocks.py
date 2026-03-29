import torch
import torch.nn as nn


class BiGRUBlock(nn.Module):
    # 2 layer Bidirectional GRU.

    def __init__(self, embed_dim, hidden_dim, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=2, #stacked GRU layers
            batch_first=True,
            bidirectional=True, #capture context from both directions
            dropout=dropout,
        )

    def forward(self, x):
        # x: (batch, seq_len, embed_dim) tensor shape of x, number of samples, number of tokens in each sample, and embedding dimension
        # output: (batch, seq_len, hidden_dim * 2) because hidden states from both directions are concatenated
        output, _ = self.gru(x) # discard final hidden state h_n
        return output


class AttentionPooling(nn.Module):
    # Learns importance scores and returns a weighted sum.

    def __init__(self, hidden_dim):
        super().__init__()
        # hidden_dim * 2 because BiGRU outputs are concatenated
        self.attention = nn.Linear(hidden_dim * 2, 1)

    def forward(self, gru_output, attention_mask):
        # gru_output: (batch, seq_len, hidden_dim * 2)
        # attention_mask: (batch, seq_len) 1 for real tokens, 0 for padding

        scores = self.attention(gru_output).squeeze(-1)  # (batch, seq_len, 1) removed last dimension to get (batch, seq_len)
        #scores for each token
        # mask padding tokens to -inf so softmax gives them zero weight
        scores = scores.masked_fill(attention_mask == 0, float('-inf')) #fill scores of padding tokens with -inf

        weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # (batch, seq_len, 1) for compatible shape in weighted sum
        # weighted sum over token dimension
        pooled = (gru_output * weights).sum(dim=1)  # (batch, hidden_dim * 2) concatenate the sum of all token * weight, control contribution to final vector
        return pooled



class NumericalBlock(nn.Module):
    #2 linear layers with ReLU for processing numerical features.

    def __init__(self, num_features, hidden_dim, dropout=0.3):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: (batch, num_features)
        return self.layers(x)  # (batch, hidden_dim)
