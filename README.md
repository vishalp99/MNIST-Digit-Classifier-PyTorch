# MNIST-Digit-Classifier-PyTorch
The project focuses on building, training, and evaluating a feedforward neural network using PyTorch to classify handwritten digits from the MNIST dataset.

---

## Project Overview

This project implements a **multiclass classification model** that recognizes digits (0–9) from 28×28 grayscale images.  
The goal is to understand foundational neural network concepts such as:

- Dataset preprocessing  
- Building fully connected neural networks  
- Activation functions (ReLU)  
- Softmax output for multiclass classification  
- Training using Adam optimizer  
- Evaluating model accuracy  
- Plotting training curves (accuracy & loss)

---

## Model Architecture

The neural network architecture used:

- **Input Layer:** 784 features (28×28 flattened)
- **Hidden Layer 1:** 128 neurons + ReLU  
- **Hidden Layer 2:** 64 neurons + ReLU  
- **Output Layer:** 10 units with Softmax activation

```python
import torch
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.model(x)


```

---

## How to Run the Project

1. Clone the repository:

```bash
git clone <repo-url>
cd MNIST-Digit-Classifier-PyTorch
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start Jupyter Notebook:

```bash
jupyter notebook
```

4. Open the .ipynb file 

---

## Results & Analysis

- The notebook includes:

- Training vs. validation accuracy plot

- Training vs. validation loss plot

- Final test accuracy

- Discussion on:

    - Number of neurons & hidden layers

    - Overfitting/underfitting

    - Improvements (dropout, batch normalization, etc.)

---

## Summary

This project demonstrates key concepts in:

- Neural network implementation with PyTorch

- Forward pass, backpropagation, and optimization

- Model evaluation and visualization

- Hyperparameter tuning

---
