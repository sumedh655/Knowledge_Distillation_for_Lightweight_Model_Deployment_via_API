# Knowledge Distillation

This project demonstrates building and deploying an end-to-end machine
learning application by training and optimizing a deep learning model,
and integrating it with a scalable web interface. The project
specifically focuses on **model compression using knowledge
distillation**, where a smaller "student" model learns from a larger
"teacher" model.

The application includes: - Training deep learning models with
PyTorch. - Knowledge Distillation for model compression. - API
development for inference. - Full-stack deployment for real-time
accessibility.

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
