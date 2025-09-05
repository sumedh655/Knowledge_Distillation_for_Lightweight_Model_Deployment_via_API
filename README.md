# Knowledge Distillation for Lightweight Model Deployment via API

This project implements Knowledge Distillation (KD) to train a lightweight student model from a teacher ResNet18 model, achieving efficient inference with minimal accuracy loss. The student model is deployed via FastAPI and Docker for real-time requests, demonstrating a production-ready, modular ML pipeline

## Features
- Teacher-student training using **Knowledge Distillation** on CIFAR-10  
- Student model achieves **86% accuracy** compared to **91% teacher accuracy** with only **5% loss**  
- **53× smaller model** for faster inference  
- **FastAPI** server for real-time JSON requests  
- **Modular design** and **Docker-based containerization** for easy deployment


## Dataset

The models are trained on the **CIFAR-10 dataset**. CIFAR-10 is a
collection of images commonly used for machine learning and computer
vision research.

-   **Number of Classes**: 10
-   **Classes**: Airplane, Automobile, Bird, Cat, Deer, Dog, Frog,
    Horse, Ship, Truck
-   **Image Size**: 32x32 pixels RGB
-   **Number of Images**: 60,000 (50,000 training and 10,000 testing)

You can find the dataset here: [CIFAR-10
Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

## 📊 Results

| Model              | Accuracy | Size Reduction |
|--------------------|----------|----------------|
| Teacher (ResNet18) | 91%      | –              |
| Student (KD)       | 86%      | 53× smaller    |
