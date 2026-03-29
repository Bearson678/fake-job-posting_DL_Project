import torch
import torch.nn as nn
from model_construction.blocks.blocks import BiGRUBlock, AttentionPooling, NumericalBlock


class FakeJobDetector(nn.Module):

    # Two-branch architecture:
    #  NLP branch: Embedding -> BiGRU -> Attention Pooling
    #  Numerical branch: 2 Linear layers with ReLU
    #  Merge: Concatenate -> Linear -> output logit

    def __init__(
        self,
        vocab_size,
        embed_dim,
        gru_hidden_dim,
        num_numerical_features,
        num_hidden_dim,
        dropout=0.3,
        pretrained_embeddings=None,
    ):
        super().__init__()

        # NLP branch
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=1)
        if pretrained_embeddings is not None: #if provided, copy to embedding layer, otherwise random init
            self.embedding.weight.data.copy_(pretrained_embeddings)

        self.bigru = BiGRUBlock(embed_dim, gru_hidden_dim, dropout)
        self.attention = AttentionPooling(gru_hidden_dim)

        # Numerical branch
        self.numerical = NumericalBlock(num_numerical_features, num_hidden_dim, dropout)

        #Merge branch
        merge_input_dim = (gru_hidden_dim * 2) + num_hidden_dim
        self.classifier = nn.Linear(merge_input_dim, 1)

    def forward(self, input_ids, attention_mask, numerical_features):
        # NLP branch
        embedded = self.embedding(input_ids)             # (batch, seq_len, embed_dim)
        gru_out = self.bigru(embedded)                   # (batch, seq_len, gru_hidden * 2)
        nlp_out = self.attention(gru_out, attention_mask) # (batch, gru_hidden * 2)

        # Numerical branch
        num_out = self.numerical(numerical_features)      # (batch, num_hidden)

        # Merge
        merged = torch.cat([nlp_out, num_out], dim=1)
        logit = self.classifier(merged).squeeze(-1)       # (batch, 1) -> (batch,)
        return logit #apply sigmoid outside the forward pass so doesnt interfere with training
