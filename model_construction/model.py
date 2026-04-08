import torch
import torch.nn as nn
from model_construction.blocks.blocks import BiGRUBlock, AttentionPooling, NumericalBlock, MultiHeadAttentionPooling
from torchvision.ops import sigmoid_focal_loss


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
        #self.attention = AttentionPooling(gru_hidden_dim)
        self.attention = MultiHeadAttentionPooling(gru_hidden_dim)

        # Numerical branch
        self.numerical = NumericalBlock(num_numerical_features, num_hidden_dim, dropout)
        
        # Auxiliary head on numerical branch (forces it to learn independently)
        self.aux_classifier = nn.Linear(num_hidden_dim, 1)

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
        
        aux_logit = self.aux_classifier(num_out).squeeze(-1)  # numerical branch only
        return logit, aux_logit
        
        return logit #apply sigmoid outside the forward pass so doesnt interfere with training
    
    
    def fit(self, dataloader, val_dataloader, num_epochs, learning_rate, save_path="best_model.pt", pos_weight=None, aux_weight=0):
        # pos_weight upweights fake job loss to improve recall
        #criterion = nn.BCELoss()
        #pos_weight = torch.tensor([pos_weight], device=self.device) if pos_weight is not None else None
        #criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=1e-3)
        
        train_losses     = []
        val_losses       = []
        best_val_loss    = float('inf')

        for epoch in range(num_epochs):
            # --- Training ---
            self.train()
            total_train_loss = 0.0

            for inputs, targets in dataloader:
                optimizer.zero_grad()
                input_ids          = inputs['input_ids']
                attention_mask     = inputs['attention_mask']
                numerical_features = inputs['numerical_features']
                targets            = targets.to(self.device).float()

                outputs, aux_outputs = self.forward(input_ids, attention_mask, numerical_features)
                #loss    = criterion(torch.sigmoid(outputs), targets)  # ✅ no sigmoid, BCEWithLogitsLoss handles it
                #loss = criterion(outputs, targets)  # no sigmoid, BCEWithLogitsLoss handles it
                
                #loss = sigmoid_focal_loss(outputs, targets, alpha=0.75, gamma=1.0, reduction='mean')
                main_loss = sigmoid_focal_loss(outputs, targets, alpha=0.75, gamma=1.0, reduction='mean')
                aux_loss  = sigmoid_focal_loss(aux_outputs, targets, alpha=0.75, gamma=1.0, reduction='mean')
                loss = main_loss + (aux_weight * aux_loss)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()
                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(dataloader)
            train_losses.append(avg_train_loss)

            # --- Validation ---
            self.eval()
            total_val_loss = 0.0

            with torch.no_grad():
                for inputs, targets in val_dataloader:
                    input_ids          = inputs['input_ids']
                    attention_mask     = inputs['attention_mask']
                    numerical_features = inputs['numerical_features']
                    targets            = targets.to(self.device).float()

                    outputs, aux_outputs = self.forward(input_ids, attention_mask, numerical_features)
                    #loss    = criterion(torch.sigmoid(outputs), targets)  # ✅ no sigmoid
                    #loss   = criterion(outputs, targets)  # no sigmoid, BCEWithLogitsLoss handles it
                    
                    #loss = sigmoid_focal_loss(outputs, targets, alpha=0.75, gamma=1.0, reduction='mean')
                    main_loss = sigmoid_focal_loss(outputs, targets, alpha=0.75, gamma=1.0, reduction='mean')
                    aux_loss  = sigmoid_focal_loss(aux_outputs, targets, alpha=0.75, gamma=1.0, reduction='mean')
                    loss = main_loss + (aux_weight * aux_loss)
                    
                    total_val_loss += loss.item()

            avg_val_loss = total_val_loss / len(val_dataloader)
            val_losses.append(avg_val_loss)

            print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

            # --- Early Stopping ---
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self.save(save_path)
                print(f"  ✅ Best model saved (val_loss={best_val_loss:.4f})")
            else:
                print(f"  ⚠️ No improvement (best_val_loss={best_val_loss:.4f})")

        self.load(save_path)
        print(f"\nRestored best model with val_loss={best_val_loss:.4f}")
        return train_losses, val_losses


    def evaluate(self, dataloader, threshold=0.3):
        from sklearn.metrics import classification_report
        
        self.eval()
        all_preds  = []
        all_labels = []

        with torch.no_grad():
            for inputs, targets in dataloader:
                input_ids          = inputs['input_ids']
                attention_mask     = inputs['attention_mask']
                numerical_features = inputs['numerical_features']
                targets            = targets.to(self.device).float()

                outputs, _ = self.forward(input_ids, attention_mask, numerical_features)
                preds   = (torch.sigmoid(outputs) >= threshold).long()  # ✅ tunable threshold

                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(targets.long().cpu().tolist())

        print(classification_report(all_labels, all_preds, target_names=['Real', 'Fake']))

    def evaluate_branches(self, dataloader, threshold=0.3):
        from sklearn.metrics import classification_report

        configs = {
            "Full model":       dict(zero_nlp=False, zero_num=False),
            "NLP only":         dict(zero_nlp=False, zero_num=True),
            "Numerical only":   dict(zero_nlp=True,  zero_num=False),
            "Both zeroed":      dict(zero_nlp=True,  zero_num=True),
        }

        for name, cfg in configs.items():
            self.eval()
            all_preds, all_labels = [], []

            with torch.no_grad():
                for inputs, targets in dataloader:
                    input_ids          = inputs['input_ids']
                    attention_mask     = inputs['attention_mask']
                    numerical_features = inputs['numerical_features'].to(self.device)
                    targets            = targets.to(self.device).float()

                    # --- Forward with ablation ---
                    embedded = self.embedding(input_ids.to(self.device))
                    gru_out  = self.bigru(embedded)
                    nlp_out  = self.attention(gru_out, attention_mask.to(self.device))
                    num_out  = self.numerical(numerical_features)

                    if cfg["zero_nlp"]:
                        nlp_out = torch.zeros_like(nlp_out)
                    if cfg["zero_num"]:
                        num_out = torch.zeros_like(num_out)

                    merged = torch.cat([nlp_out, num_out], dim=1)
                    logit  = self.classifier(merged).squeeze(-1)
                    preds  = (torch.sigmoid(logit) >= threshold).long()

                    all_preds.extend(preds.cpu().tolist())
                    all_labels.extend(targets.long().cpu().tolist())

            print(f"\n{'='*40}\n{name}")
            print(classification_report(all_labels, all_preds, target_names=['Real', 'Fake']))


    def save(self,path):
        ### Implement model saving logic here
        torch.save(self.state_dict(), path)
        return
    
    def load(self,path):
        ### Implement model loading logic here
        self.load_state_dict(torch.load(path,map_location=self.device))
        self.to(self.device) #load to cpu by default, can be modified to load to gpu if needed
        return
    
