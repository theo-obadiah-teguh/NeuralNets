import torch
import utils

class ImageClassifierTrainer():
    def __init__(self, model, optimizer, loss_func, max_epochs, train_loader, valid_loader, print_freq):
        """
        This method initializes a trainer class.
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.max_epochs = max_epochs
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        
    def enable_parallel_processing(self):
        """
        This method turns on GPU processing if available on the local device.
        """
            
        if torch.cuda.is_available() and torch.cuda.device_count() > 1: # Check for multiple CUDA GPUs
            device = torch.device("cuda")
            self.model = torch.nn.DataParallel(self.model.to(device))
            print(f"Using {torch.cuda.device_count()} CUDA GPUs.")

        elif torch.cuda.is_available(): # Check for a single CUDA GPU
            device = torch.device("cuda")
            self.model = self.model.to(device)
            print("Using a Single CUDA GPU.")

        elif torch.backends.mps.is_available(): # Check for MPS (Metal Performance Shaders) on M1
            device = torch.device("mps")
            self.model = self.model.to(device)
            print("Using MPS.")

        else:
            device = torch.device("cpu")
            self.model = self.model.to(device)
            print("No Parallel Option: Using CPU.")

    def train_epoch(self, epoch):
        """
        This method defines a single epoch (forward -> backward -> gradient descent) for the WHOLE dataset.
        Note: The model should be in training mode, so that Batch-Normalization and Dropout layers are updated.
        """
        losses = utils.AverageMeter()
        top1 = utils.AverageMeter()
        
        self.model.train()
        
        # Get the current device we are using (CPU or GPU or MPS)
        device = next(self.model.parameters()).device
        
        # Each iteration is equal to a TRAINING STEP, where we process ONE BATCH
        for idx, (inputs, labels) in enumerate(self.train_loader):
            # Note that the trainloader consists of [idx, (input, label)] "rows"
            # Make predictions, and calculate loss (forward prop)
            outputs = self.model(inputs.to(device))
            loss = self.loss_func(outputs, labels.to(device))

            # This next line is an implementation detail
            # Empty the gradient buffers, so gradients won't be accumulated to existing gradients
            self.model.zero_grad()
    
            # Backprop the calculated gradients to each node, the gradients are functions of loss
            loss.backward()
    
            # Gradient descent: New Weights = Old Weights - Learning Rate * Gradient
            # Update the weights in each layer
            self.optimizer.step()

            # Calculate precision
            prec1 = utils.top_k_accuracy(outputs, labels.to(device), k=1)

            # Update perfomance trackers
            # Note that the moving avg should be proportional to the size of the batch
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))

            if idx % self.print_freq == 0:
                # Print the training batch idx and performance metrics
                print(f'Train: [{epoch}, {idx}/{len(self.train_loader)}]\t'
                      f'Loss {losses.value:.4f} ({losses.avg:.4f})\t'
                      f'Batch Top@1 Accuracy {top1.value:.3f} ({top1.avg:.3f})')

    def validate(self):
        """
        This method assesses model performance, i.e. it goes through the whole dataset.
        Note: The model should be in evaluation mode, so that Batch-Normalization and Dropout layers aren't updated.
        """
        losses = utils.AverageMeter()
        top1 = utils.AverageMeter()
        
        self.model.eval()
        
        # Get the current device we are using (CPU or GPU or MPS)
        device = next(self.model.parameters()).device

        # Disable gradient computation, because Autograd generates gradients in a forward pass
        # This should speed up computation
        with torch.no_grad():
            for idx, (inputs, labels) in enumerate(self.valid_loader):
                # Note that the trainloader consists of [idx, (input, label)] "rows"
                # Make predictions, and calculate loss (forward prop)
                outputs = self.model(inputs.to(device))
                loss = self.loss_func(outputs, labels.to(device))
    
                # Calculate precision
                prec1 = utils.top_k_accuracy(outputs, labels.to(device), k=1)
    
                # Update perfomance trackers
                # Note that the moving avg should be proportional to the size of the batch
                losses.update(loss.item(), inputs.size(0))
                top1.update(prec1.item(), inputs.size(0))
    
                if idx % self.print_freq == 0:
                    # Print the testing batch idx, performance metrics and their moving averages
                    print(f'Valid: [{idx}/{len(self.valid_loader)}]\t'
                          f'Loss {losses.value:.4f} ({losses.avg:.4f})\t'
                          f'Batch Top@1 Accuracy {top1.value:.3f} ({top1.avg:.3f})')

        return top1.avg # Average accuracy of across whole testing dataset

    def fit(self):
        """
        Full training loop, with validation for every epoch of training.
        """
        self.enable_parallel_processing()
        
        for epoch in range(1, self.max_epochs + 1):
            self.train_epoch(epoch)
            prec1 = self.validate()

