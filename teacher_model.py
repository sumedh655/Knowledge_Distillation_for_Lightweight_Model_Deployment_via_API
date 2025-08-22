import torch
import torch.nn as nn
import torchvision.models as models

class TeacherModel(nn.Module):
    """
    ResNet-18 based teacher model for CIFAR-10
    High capacity model with 11.2M parameters
    """
    def __init__(self, num_classes=10):
        super(TeacherModel, self).__init__()
        # Load pre-trained ResNet-18 and modify for CIFAR-10
        self.model = models.resnet18(pretrained=False)

        # Modify the first conv layer for 32x32 input
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                                   stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()  # Remove maxpool for small images

        # Modify final layer for CIFAR-10 (10 classes)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

def create_teacher_model(device='cuda'):
    """Create and initialize teacher model"""
    model = TeacherModel(num_classes=10)
    model = model.to(device)
    return model

# Training configuration
def get_teacher_config():
    return {
        'lr': 0.1,
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'epochs': 200,
        'batch_size': 128,
        'scheduler': 'cosine'
    }
