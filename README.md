# handwritten Dzongkha digit detection using ML

This project uses a **ResNet18** model to classify handwritten digits (0-9) using the Dzongkha digit dataset.  
The model uses **data augmentation** and **transfer learning techniques** to improve generalization and performance.
Achieved Test Accuracy: <span style="font-size:30px; font-weight:bold;">98.5%</span>

A deep learning model that classifies handwritten Dzongkha digits (0-9) using PyTorch and ResNet-18 architecture.

## Features

- **Model Architecture**: Modified ResNet-18 adapted for grayscale images
- **Data Augmentation**: Random rotation, scaling, and cropping during training
- **Training Pipeline**: Includes validation and model checkpointing
- **Evaluation**: Provides classification report and confusion matrix
- **Interactive GUI**: Draw digits and get real-time predictions
- **Interactive Canvas**: Draw digits with your mouse
- **Real-time Prediction**: Get instant classification results
- **Confidence Visualization**: See prediction probabilities for all digits

## Requirements

- Python 3.7+
- PyTorch
- torchvision
- scikit-learn
- matplotlib
- seaborn
- pillow
- tqdm
- tkinter (usually included with Python)

## Dataset

- kaggle handwritten dzongkha digit dataset
- ![dataset](image-2.png)

## Results

- After training the model, we obtained the following results on the test dataset:
- Test Accuracy: <span style="font-size:30px; font-weight:bold;">98.5%</span>
![Accuracy](image.png)
![confusion matrix](image-1.png)

## checking on canvas the model

![predicted-1](image-3.png)
![predicted-0](image-4.png)
![predicted-6](image-5.png)
![predicted-7](image-6.png)
![predicted-8](image-7.png)
![predicted-5](image-8.png)
![predicted-3](image-9.png)
![predicted-2](image-10.png)
![predicted-4](image-11.png)
![predicted-9](image-13.png)