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
        device='cpu'
    ):
        super().__init__()
        self.device = torch.device(device)

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
        
        self.to(self.device)
        

    def forward(self, input_ids, attention_mask, numerical_features):
        # Prep inputs to device
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        numerical_features = numerical_features.to(self.device)
        
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
    
    
    def train(self,dataloader,num_epochs,learning_rate):
        ## Implements training loop for the model, Returns list of epoch losses for monitoring
        
        ## Loss calculation and optimization logic using Binary Cross-Entropy Loss and Adam optimizer.
        
        ## This follows a simple binary classification training loop structure, 
        ## iterating over epochs and batches, computing loss, backpropagating, and updating weights.
        
        ## The NLP processing are done instead in the forward pass.
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        epoch_losses = []
        for epoch in range(num_epochs):
            total_loss = 0.0
            optimizer.zero_grad()
            for inputs,targets in dataloader:
                input_ids = inputs['input_ids']
                attention_mask = inputs['attention_mask']
                numerical_features = inputs['numerical_features']
                targets = targets.to(self.device).float() #ensure targets are float for BCELoss
                outputs = self.forward(input_ids, attention_mask, numerical_features)
                loss = criterion(torch.sigmoid(outputs), targets) #apply sigmoid to outputs for BCELoss
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(dataloader)
            epoch_losses.append(avg_loss)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
        return epoch_losses
    
    
    def save(self,path):
        ### Implement model saving logic here
        torch.save(self.state_dict(), path)
        return
    
    def load(self,path):
        ### Implement model loading logic here
        self.load_state_dict(torch.load(path,map_location=self.device))
        self.to(self.device) #load to cpu by default, can be modified to load to gpu if needed
        return