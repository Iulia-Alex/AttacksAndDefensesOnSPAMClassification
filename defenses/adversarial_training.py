import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class AdversarialTraining:
    """
    Improved adversarial training with dynamic mixing
    """
    def __init__(self, model, attack_method, mix_ratio=0.5):
        self.model = model
        self.attack_method = attack_method
        self.mix_ratio = mix_ratio
    
    def train_epoch(self, train_loader, optimizer, device):
        """Train one epoch with adversarial examples"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        criterion = nn.CrossEntropyLoss()
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            batch_size = data.size(0)
            
            # Generate adversarial examples for entire batch
            adv_embeddings = self.attack_method.attack(data, target)
            
            # Forward pass for clean examples
            optimizer.zero_grad()
            clean_outputs = self.model(data)
            
            # Forward pass for adversarial examples
            lstm_out, (hidden, cell) = self.model.lstm(adv_embeddings)
            hidden_concat = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
            hidden_concat = self.model.dropout(hidden_concat)
            hidden_concat = F.relu(self.model.fc1(hidden_concat))
            hidden_concat = self.model.bn(hidden_concat)
            hidden_concat = self.model.dropout(hidden_concat)
            adv_outputs = self.model.fc2(hidden_concat)
            
            # Calculate losses
            clean_loss = criterion(clean_outputs, target)
            adv_loss = criterion(adv_outputs, target)
            
            # Combined loss with mixing ratio
            loss = (1 - self.mix_ratio) * clean_loss + self.mix_ratio * adv_loss
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            
            # Calculate accuracy (on clean examples)
            _, predicted = clean_outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        
        return total_loss / len(train_loader), 100. * correct / total


def train_with_adversarial_training(data, base_model, epochs=20, mix_ratio=0.6, epsilon=0.1):
    """
    Improved adversarial training with better parameters
    """
    from attacks.pgd import PGDAttack
    from models.baseline_model import SpamClassifier
    from torch.utils.data import TensorDataset, DataLoader
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Clone model
    vocab_size = len(data['preprocessor'].word2idx)
    defended_model = SpamClassifier(vocab_size).to(device)
    defended_model.load_state_dict(base_model.state_dict())
    
    # Setup attack with shorter iterations for efficiency
    pgd_attack = PGDAttack(defended_model, epsilon=epsilon, alpha=epsilon/10, num_iter=10)
    
    # Setup adversarial training
    adv_trainer = AdversarialTraining(defended_model, pgd_attack, mix_ratio=mix_ratio)
    
    # Prepare data loaders
    train_dataset = TensorDataset(
        torch.LongTensor(data['X_train']),
        torch.LongTensor(data['y_train'])
    )
    val_dataset = TensorDataset(
        torch.LongTensor(data['X_val']),
        torch.LongTensor(data['y_val'])
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Use lower learning rate for fine-tuning
    optimizer = optim.Adam(defended_model.parameters(), lr=0.00005)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                                      factor=0.5, patience=3, verbose=True)
    
    best_val_acc = 0
    best_model_state = None
    patience_counter = 0
    patience = 5
    
    print(f"Training with Adversarial Training (mix_ratio={mix_ratio}, eps={epsilon})...")
    
    for epoch in range(epochs):
        # Training
        train_loss, train_acc = adv_trainer.train_epoch(train_loader, optimizer, device)
        
        # Validation
        defended_model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = defended_model(inputs)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        
        # Learning rate scheduling
        scheduler.step(val_acc)
        
        # Early stopping based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = defended_model.state_dict().copy()
            patience_counter = 0
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                  f"Val Acc: {val_acc:.2f}% [DONE] BEST")
        else:
            patience_counter += 1
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                  f"Val Acc: {val_acc:.2f}% (Patience: {patience_counter}/{patience})")
            
            if patience_counter >= patience:
                print(f"Early stopping triggered!")
                break
    
    # Load best model
    if best_model_state is not None:
        defended_model.load_state_dict(best_model_state)
        print(f"Loaded best model with validation accuracy: {best_val_acc:.2f}%")
    
    # Save defended model
    torch.save(defended_model.state_dict(), 'models/adversarial_training_model.pth')
    print("Adversarial training complete!")
    
    return defended_model