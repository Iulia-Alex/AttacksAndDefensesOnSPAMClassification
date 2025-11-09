import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from models.baseline_model import SpamClassifier

def train_with_adversarial_training(data, base_model, epochs=20):
    """
    Train with adversarial training defense (FIXED VERSION)
    """
    from attacks.pgd import PGDAttack
    from models.baseline_model import SpamClassifier
    from torch.utils.data import TensorDataset, DataLoader
    import torch
    import torch.optim as optim
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Clone model
    vocab_size = len(data['preprocessor'].word2idx)
    defended_model = SpamClassifier(vocab_size).to(device)
    defended_model.load_state_dict(base_model.state_dict())
    
    # Setup attack
    pgd_attack = PGDAttack(defended_model, epsilon=0.1, alpha=0.01, num_iter=10)
    
    # Setup adversarial training
    from defenses.adversarial_training import AdversarialTraining
    adv_trainer = AdversarialTraining(defended_model, pgd_attack, mix_ratio=0.5)
    
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
    
    optimizer = optim.Adam(defended_model.parameters(), lr=0.00005)
    
    best_val_acc = 0
    best_model_state = None
    patience_counter = 0
    patience = 5
    
    print(f"Training with Adversarial Training defense...")
    
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
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = defended_model.state_dict().copy()
            patience_counter = 0
            print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f}, "
                  f"Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}% [DONE] BEST")
        else:
            patience_counter += 1
            print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f}, "
                  f"Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}% "
                  f"(Patience: {patience_counter}/{patience})")
            
            if patience_counter >= patience:
                print(f"Early stopping triggered!")
                break
    
    # Load best model
    if best_model_state is not None:
        defended_model.load_state_dict(best_model_state)
    
    # Save defended model
    torch.save(defended_model.state_dict(), 'models/adversarial_training_model.pth')
    print("Adversarial training complete!")
    
    return defended_model

def train_with_defensive_distillation(data, teacher_model, epochs=20, temperature=20):
    """
    Two-stage defensive distillation training
    Stage 1: Train teacher with high temperature
    Stage 2: Distill to student model
    """
    from defenses.defensive_distillation import DefensiveDistillation
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Prepare data loader
    train_dataset = TensorDataset(
        torch.LongTensor(data['X_train']),
        torch.LongTensor(data['y_train'])
    )
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    print(f"=== STAGE 1: Re-train teacher with temperature {temperature} ===")
    # Stage 1: Re-train teacher model with temperature
    vocab_size = len(data['preprocessor'].word2idx)
    teacher_temp = SpamClassifier(vocab_size).to(device)
    teacher_temp.load_state_dict(teacher_model.state_dict())
    
    optimizer_teacher = optim.Adam(teacher_temp.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs // 2):  # Train for half the epochs
        teacher_temp.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for data_batch, target in train_loader:
            data_batch, target = data_batch.to(device), target.to(device)
            
            optimizer_teacher.zero_grad()
            # Apply temperature during training
            logits = teacher_temp(data_batch) / temperature
            loss = criterion(logits, target)
            loss.backward()
            optimizer_teacher.step()
            
            total_loss += loss.item()
            # Predictions without temperature for accuracy
            with torch.no_grad():
                pred_logits = teacher_temp(data_batch)
                _, predicted = pred_logits.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        acc = 100. * correct / total
        avg_loss = total_loss / len(train_loader)
        print(f"Teacher Epoch {epoch+1}/{epochs//2} - Loss: {avg_loss:.4f}, Acc: {acc:.2f}%")
    
    print(f"\n=== STAGE 2: Distill to student model ===")
    # Stage 2: Create and train student model
    student_model = SpamClassifier(vocab_size).to(device)
    distillation = DefensiveDistillation(teacher_temp, student_model, temperature)
    
    optimizer_student = optim.Adam(student_model.parameters(), lr=0.0001)
    
    for epoch in range(epochs):
        loss, acc, distill_loss, student_loss = distillation.train_epoch(
            train_loader, optimizer_student, device, alpha=0.9
        )
        print(f"Student Epoch {epoch+1}/{epochs} - "
              f"Total Loss: {loss:.4f}, Acc: {acc:.2f}%, "
              f"Distill: {distill_loss:.4f}, Student: {student_loss:.4f}")
    
    # Save defended model
    torch.save(student_model.state_dict(), 'models/defensive_distillation_model.pth')
    print("Defensive distillation complete!")
    
    return student_model