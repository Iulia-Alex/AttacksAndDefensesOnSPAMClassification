import torch
import torch.nn as nn
import torch.nn.functional as F

class SpamClassifier(nn.Module):
    """
    LSTM-based text classifier with regularization
    """
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2, dropout=0.5):
        super(SpamClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout, bidirectional=True)
        self.fc1 = nn.Linear(hidden_dim * 2, 128)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, 2)
        
        # Add batch normalization for better training
        self.bn = nn.BatchNorm1d(128)
        
    def forward(self, x, return_embeddings=False):
        embedded = self.embedding(x)
        
        if return_embeddings:
            return embedded
        
        lstm_out, (hidden, cell) = self.lstm(embedded)
        hidden_concat = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        
        x = self.dropout(hidden_concat)
        x = F.relu(self.fc1(x))
        x = self.bn(x)  # Batch normalization
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits