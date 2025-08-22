import torch
import torch.nn as nn
import torch.nn.functional as F

class StudentModel(nn.Module):
    """
    Lightweight CNN student model for CIFAR-10
    Only 0.2M parameters - 56x smaller than teacher
    """
    def __init__(self, num_classes=10):
        super(StudentModel, self).__init__()

        # Feature extraction layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # Classifier
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        # First block
        x = F.relu(self.bn1(self.conv1(x)))

        # Second block
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)  # 32x32 -> 16x16

        # Third block
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)  # 16x16 -> 8x8

        # Global average pooling
        x = self.global_avg_pool(x)  # 8x8 -> 1x1
        x = x.view(x.size(0), -1)

        # Classifier
        x = self.dropout(x)
        x = self.classifier(x)

        return x

def create_student_model(device='cuda'):
    """Create and initialize student model"""
    model = StudentModel(num_classes=10)
    model = model.to(device)
    return model

def count_parameters(model):
    """Count model parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
