import torch
import torch.nn as nn
import torch.nn.functional as F

class PGDAttack:
    """
    Projected Gradient Descent attack on text embeddings
    """
    def __init__(self, model, epsilon=0.1, alpha=0.01, num_iter=40):
        """
        Args:
            model: Target model
            epsilon: Maximum perturbation budget (L-infinity norm)
            alpha: Step size
            num_iter: Number of iterations
        """
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_iter = num_iter
        
    def attack(self, x, y):
        """
        Generate adversarial examples using PGD
        
        Args:
            x: Input sequences [batch_size, seq_len]
            y: True labels [batch_size]
        
        Returns:
            Adversarial embeddings
        """
        self.model.eval()
        
        # Get original embeddings
        with torch.no_grad():
            orig_embeddings = self.model.embedding(x)
        
        # Initialize adversarial embeddings
        adv_embeddings = orig_embeddings.detach().clone()
        adv_embeddings.requires_grad = True
        
        for i in range(self.num_iter):
            # Forward pass through LSTM and classifier (skip embedding layer)
            lstm_out, (hidden, cell) = self.model.lstm(adv_embeddings)
            hidden_concat = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
            hidden_concat = self.model.dropout(hidden_concat)
            hidden_concat = F.relu(self.model.fc1(hidden_concat))
            hidden_concat = self.model.dropout(hidden_concat)
            logits = self.model.fc2(hidden_concat)
            
            # Calculate loss (maximize loss for true class)
            loss = nn.CrossEntropyLoss()(logits, y)
            
            # Get gradients
            self.model.zero_grad()
            loss.backward()
            grad = adv_embeddings.grad.data
            
            # Update adversarial embeddings
            with torch.no_grad():
                # Sign of gradient (FGSM step)
                perturbation = self.alpha * grad.sign()
                adv_embeddings = adv_embeddings + perturbation
                
                # Project back to epsilon ball
                delta = adv_embeddings - orig_embeddings
                delta = torch.clamp(delta, -self.epsilon, self.epsilon)
                adv_embeddings = orig_embeddings + delta
                
            adv_embeddings = adv_embeddings.detach()
            adv_embeddings.requires_grad = True
        
        return adv_embeddings.detach()