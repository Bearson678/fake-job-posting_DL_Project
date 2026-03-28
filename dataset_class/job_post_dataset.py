import torch
from torch.utils.data import Dataset, DataLoader

class JobPostingDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=2048):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        # Longformer specific: Global Attention Mask
        # We set global attention on the <s> token (index 0)
        input_ids = encoding['input_ids'].flatten()
        attention_mask = encoding['attention_mask'].flatten()
        global_attention_mask = torch.zeros_like(attention_mask)
        global_attention_mask[0] = 1

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'global_attention_mask': global_attention_mask,
            'labels': torch.tensor(label, dtype=torch.long)
        }

