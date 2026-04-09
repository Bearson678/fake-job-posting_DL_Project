import torch
from torch.utils.data import Dataset, DataLoader

class JobPostingDataset(Dataset):
    def __init__(self, texts_tok, numerical_features, labels, max_len=2048):
        self.texts_tok           = texts_tok   # already tokenized integer sequences
        self.numerical_features  = numerical_features  # (N, num_features) array or dataframe
        self.labels              = labels
        self.max_len             = max_len

    def __len__(self):
        return len(self.texts_tok)

    def __getitem__(self, idx):
        tokens = self.texts_tok[idx][:self.max_len]
        label  = self.labels[idx]

        # Pad to max_len
        pad_len        = self.max_len - len(tokens)
        attention_mask = [1] * len(tokens) + [0] * pad_len
        tokens         = tokens + [1] * pad_len   # 1 = <pad> index

        inputs = {
            'input_ids':           torch.tensor(tokens,         dtype=torch.long),
            'attention_mask':      torch.tensor(attention_mask, dtype=torch.long),
            'numerical_features':  torch.tensor(self.numerical_features[idx], dtype=torch.float),
        }
        label = torch.tensor(label, dtype=torch.float)  # float for BCELoss
        return inputs, label
    