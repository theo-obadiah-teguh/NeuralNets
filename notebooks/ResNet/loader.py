from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader

class CIFAR10DataLoader():
    def __init__(self, batch_size=128):
        """
        This method creates a DataLoader class and defines the transformations.
        """
        self.batch_size = batch_size # Optional instance variable for training batch size
        
        # Not in the original ResNet paper: standard ImageNet normalization
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], # These values represent RGB channels
                                              std=[0.229, 0.224, 0.225])

        # Use the transformation defined in page 7, Section 4.2, para. 3
        # Note, the we pad with zeroes (black) by default
        self.train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally
            transforms.RandomCrop(32, 4),       # Randomly crop 32x32 image after padding 4 pixels on each side
            transforms.ToTensor(),              # Convert PIL Image to Tensor
            self.normalize,                     # Normalize the image
        ])
        
        self.valid_transform = transforms.Compose([
            transforms.ToTensor(),              # Convert PIL Image to Tensor
            self.normalize,                     # Normalize the image
        ])

    def setup_datasets(self):
        """
        This method downloads the dataset and performs the necessary transformations.
        Data loading is separate from __init__() so a downloads don't occur every DataLoader class instantiation.
        """
        # Load the data and perform the transformation
        # We download the data if not yet available, and get the 50,000-element training data
        self.train_dataset = datasets.CIFAR10(root='./data', # location for saving when downloaded
                                              train=True,
                                              download=True,
                                              transform=self.train_transform)

        # Data should already be downloaded, and we now just get the 10,000-element validation data
        self.valid_dataset = datasets.CIFAR10(root='./data', # location for saving when downloaded
                                              train=False,
                                              download=False,
                                              transform=self.valid_transform)

    def get_train_loader(self):
        """
        Returns a training set dataloader. Shuffling is used to prevent learning order bias.
        Without shuffling, the model would see the training samples in the exact same order in every epoch.
        Shuffling ensures that each batch in each epoch contains a random subset of the training data.
        """
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def get_valid_loader(self):
        """
        Returns a training set dataloader. Shuffling is not used to allow reproducibility of performance metrics.
        """
        return DataLoader(self.valid_dataset, batch_size=128, shuffle=False)

    def get_loaders(self):
        """
        Returns both training and validation loaders in a tuple for easy unpacking.
        """
        return self.get_train_loader(self), self.get_valid_loader(self)

    