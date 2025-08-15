# NeuralNets

A repository containing ANN implementations with PyTorch, starting out with the basics all the way to ResNet.

## Project Overview

This project was supervised by Prof. Xiaolin Huang and his research group at Shanghai Jiaotong University. The goal was to design a ResNet implementation that is comparable with the [original paper](https://arxiv.org/abs/1512.03385).

That said, the code utilized in this repository was deeply inspired by [akamaster's repository](https://github.com/akamaster/pytorch_resnet_cifar10). In fact, this repository was an attempt at refactoring existing code from the aforementioned repository, using an object-oriented approach and newer PyTorch based functions.

In addition to this, we aim to add sufficient documentation to this repository, such that it would be easier for the AI beginner to digest. For example, the ResNet class is implemented in a notebook containing detailed notes about the implementation. Furthermore, one may look into some relevant fundamental tutorials from [StatQuest Introduction to Neural Networks and AI](https://github.com/StatQuest/signa).

## Pretrained Models

Similar to `akamaster`'s implementation, we also provide pretrained-models for each ResNet subclass. These models were trained with online Kaggle resources, specifically a Tesla P100 GPU.

| Model Name  | Parameters | Top-1 Accuracy |
|-------------|------------|----------------|
| resnet-20   | 0.27M      | **91.47%**     |
| resnet-32   | 0.46M      | **92.32%**     |
| resnet-44   | 0.66M      | **92.78%**     |
| resnet-56   | 0.85M      | **93.10%**     |
| resnet-110  | 1.7M       | **_____%**     |
| resnet-1202 | 19.4M      | **_____%**     |

## Relevant Docker Tutorials

1. [Fireship's Docker Tutorial](https://www.youtube.com/watch?v=gAkwW2tuIqE)
2. [Jupyter Notebook and Docker Tutorial](https://www.youtube.com/watch?v=8qcCkifuq0E)
3. [Jupyter Setup with Docker Volume](https://www.youtube.com/watch?v=ajPppaAVXQU)
