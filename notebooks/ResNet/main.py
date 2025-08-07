"""
This is the main script used to train our ResNet architecture.
"""

# Load Custom Modules
import import_ipynb # Use ResNet straight from the notebook
import resnet 
from trainer import ImageClassifierTrainer
from loader import CIFAR10DataLoader

# Load PyTorch Modules
import torch
import torch.optim as optim # For optimizer
import torch.nn as nn # For loss function

def main():
    # --- Configuration (Hardcoded for MVP) ---
    model_name = 'resnet32'
    batch_size = 128
    epochs = 10 # Reduced for quick MVP test
    learning_rate = 0.1
    momentum = 0.9
    weight_decay = 1e-4

    # --- 1. Model ---
    print(f"Initializing model: {model_name}")
    model = resnet.__dict__[model_name]() # Create the ResNet model
    # Note: The original uses DataParallel, omitted for simplicity in MVP
    # model = torch.nn.DataParallel(model)

    # --- 2. Data ---
    print("Setting up data loaders...")
    data_loader = CIFAR10DataLoader(batch_size=batch_size)
    data_loader.setup_datasets()
    train_loader, valid_loader = data_loader.get_loaders()

    # --- 3. Loss, Optimizer ---
    print("Setting up loss and optimizer...")
    # Assuming CUDA/MPS/CPU handling is inside the trainer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), learning_rate,
                          momentum=momentum, weight_decay=weight_decay)
    # Note: Learning rate scheduler omitted for MVP

    # --- 4. Trainer ---
    print("Initializing trainer...")
    trainer = ImageClassifierTrainer(model, optimizer, criterion, epochs, train_loader, valid_loader)

    # --- 5. Train ---
    print("Starting fit process...")
    trainer.fit()

    print("MVP Training Completed.")

if __name__ == '__main__':
    main()