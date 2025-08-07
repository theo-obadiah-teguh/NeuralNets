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

# Load Utilities
import argparse

# Define an argument parser
parser = argparse.ArgumentParser(description='CIFAR-10 ResNet Trainer')
parser.add_argument('-a', '--aarch', dest='model_name', default='resnet32')
parser.add_argument('-e', '--epochs', dest='epochs', default=10)
parser.add_argument('-m', '--momentum', dest='momentum', default=0.9)
parser.add_argument('-w', '--workers', dest='workers', default=2)
parser.add_argument('-pf', '--print-freq', dest='print_freq', default=50)
parser.add_argument('-bs', '--batch-size', dest='batch_size', default=128)
parser.add_argument('-wd', '--weight-decay', dest='weight_decay', default=1e-4)
parser.add_argument('-lr', '--learning-rate', dest='learning_rate', default=0.1)

# Get all arguments from the command line
args = parser.parse_args()

def main():
    # --- 1. Model ---
    print(f"Initializing model: {args.model_name}")
    model = resnet.__dict__[args.model_name]() # Create the ResNet model

    # --- 2. Data ---
    data_loader = CIFAR10DataLoader(batch_size=args.batch_size, num_workers=args.workers)
    data_loader.setup_datasets()
    train_loader, valid_loader = data_loader.get_loaders()

    # --- 3. Loss, Optimizer ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),
                          args.learning_rate,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)
    
    # Note: Learning rate scheduler omitted for MVP

    # --- 4. Training
    trainer = ImageClassifierTrainer(model, 
    optimizer, criterion, args.epochs, train_loader, valid_loader, args.print_freq)
    trainer.fit()

    print("MVP Training Completed.")

if __name__ == '__main__':
    main()