import torch
import torch.nn as nn
import torch.nn.functional as F

class DefensiveDistillation:
    """
    Improved Defensive Distillation with two-stage training
    """
    def __init__(self, teacher_model, student_model, temperature=20):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.temperature = temperature
    
    def distillation_loss(self, student_logits, teacher_logits, labels, alpha=0.9):
        """
        Calculate distillation loss with higher weight on soft targets
        """
        # Soften probabilities with temperature
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
        
        # KL divergence loss for soft targets
        distillation_loss = nn.KLDivLoss(reduction='batchmean')(
            soft_student, soft_teacher
        ) * (self.temperature ** 2)
        
        # Cross-entropy loss for hard targets
        student_loss = nn.CrossEntropyLoss()(student_logits, labels)
        
        # Higher weight on distillation (changed from 0.7 to 0.9)
        total_loss = alpha * distillation_loss + (1 - alpha) * student_loss
        return total_loss, distillation_loss.item(), student_loss.item()
    
    def train_epoch(self, train_loader, optimizer, device, alpha=0.9):
        """Train student model for one epoch"""
        self.teacher_model.eval()
        self.student_model.train()
        total_loss = 0
        total_distill_loss = 0
        total_student_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # Get teacher predictions (with temperature)
            with torch.no_grad():
                teacher_logits = self.teacher_model(data) / self.temperature
            
            # Get student predictions
            optimizer.zero_grad()
            student_logits = self.student_model(data)
            
            # Calculate distillation loss
            loss, distill_loss, student_loss = self.distillation_loss(
                student_logits, teacher_logits * self.temperature, target, alpha
            )
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            total_distill_loss += distill_loss
            total_student_loss += student_loss
            _, predicted = student_logits.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        avg_distill = total_distill_loss / len(train_loader)
        avg_student = total_student_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy, avg_distill, avg_student