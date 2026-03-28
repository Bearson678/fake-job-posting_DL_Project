import torch
from torch.utils.data import Dataset, DataLoader

class JobPostingDataset(Dataset):
    def __init__(self, texts_tok, labels, max_len=2048):
        self.texts_tok = texts_tok   # already tokenized integer sequences
        self.labels    = labels
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts_tok)

    def __getitem__(self, idx):
        tokens = self.texts_tok[idx][:self.max_len]   # truncate to max_len
        label  = self.labels[idx]

        # Pad or truncate to max_len
        pad_len = self.max_len - len(tokens)
        attention_mask = [1] * len(tokens) + [0] * pad_len
        tokens         = tokens + [1] * pad_len   # 1 = <pad> index

        return {
            'input_ids':      torch.tensor(tokens,         dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels':         torch.tensor(label,          dtype=torch.long)
        }