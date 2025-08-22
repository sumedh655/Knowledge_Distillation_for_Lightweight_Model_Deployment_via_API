import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillationLoss(nn.Module):
    """
    Knowledge Distillation Loss combining:
    - KL Divergence loss (soft targets from teacher)
    - Cross Entropy loss (hard labels from dataset)
    """
    def __init__(self, alpha=0.7, temperature=4.0):
        super(DistillationLoss, self).__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, labels):
        # Soft targets (knowledge distillation loss)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=1)
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=1)

        distillation_loss = self.kl_div(student_log_probs, teacher_probs) * (self.temperature ** 2)

        # Hard targets (standard classification loss)
        classification_loss = self.ce_loss(student_logits, labels)

        # Combined loss
        total_loss = (self.alpha * distillation_loss +
                     (1 - self.alpha) * classification_loss)

        return total_loss, distillation_loss, classification_loss

def distill_knowledge(teacher_model, student_model, train_loader,
                     val_loader, device='cuda', epochs=100):
    """
    Main knowledge distillation training loop
    """
    # Set teacher to evaluation mode (frozen)
    teacher_model.eval()

    # Initialize student optimizer and loss
    optimizer = torch.optim.Adam(student_model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    criterion = DistillationLoss(alpha=0.7, temperature=4.0)

    # Training history
    train_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        # Training phase
        student_model.train()
        epoch_loss = 0.0

        for batch_idx, (data, labels) in enumerate(train_loader):
            data, labels = data.to(device), labels.to(device)

            # Forward pass through both models
            with torch.no_grad():
                teacher_logits = teacher_model(data)

            student_logits = student_model(data)

            # Calculate distillation loss
            loss, kd_loss, ce_loss = criterion(student_logits, teacher_logits, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()

        # Validation
        val_acc = evaluate_model(student_model, val_loader, device)

        train_losses.append(epoch_loss / len(train_loader))
        val_accuracies.append(val_acc)

        if epoch % 10 == 0:
            print(f'Epoch {epoch}: Loss={epoch_loss:.4f}, Val Acc={val_acc:.4f}')

    return train_losses, val_accuracies

def evaluate_model(model, data_loader, device):
    """Evaluate model accuracy"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, labels in data_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total
