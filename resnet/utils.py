import torch

class AverageMeter():
    """
    Helper class to track performance and moving averages.
    """
    def __init__(self):
        """
        This method initializes the class and its instance variables.
        """
        self.reset()
        
    def reset(self):
        """
        This method zeroes out all the instance variables.
        """
        self.value = 0
        self.count = 0
        self.sum = 0
        self.avg = 0

    def update(self, value, n=1):
        """
        This method updates the moving average, with an optional weight.
        """
        self.value = value
        self.count += n
        self.sum += value * n
        self.avg = self.sum / self.count


def top_k_accuracy(output, target, k):
    """
    Helper function to calculate top-k accuracy. (Base behavior for training loop)
    We check if the true label is present in the top-k predictions within the output vector.
    Outputs are tensors of the format [batch_size, classes=10] (matrix with rows and columns).
    Targets are tensors of the format [batch_size] (a column vector containing true classes).
    Accuracy is returned as a tensor with a single value (0-Dimensional).
    
    Examples: 
    If the true label is class 1, the output is [1.5, 0.0, 2.3], the top-2 is classes [2, 0]; this will be a miss.
    If the true label is class 1, the output is [1.5, 2.3, 2.2], the top-2 is classes [1, 2]; this will be a hit.
    """
    # By default this gives you values, indices but here we just need the indices i.e. classes
    _, indices = torch.topk(output, k, dim=1) # We get top-k across classes (dim-1)
    # Indices is now a matrix of shape (batch_size x k)

    # Check if target is in top-k predictions for each sample
    # Note that the target/label is just 0-9
    # target.unsqueeze(1) converts [batch_size] to [batch_size, 1] for broadcasting
    # torch.eq() performs element-wise equality comparison between two tensors.
    correct = torch.eq(indices, target.unsqueeze(1)).any(dim=1)
    # Correct is now a 1-D tensor of True or False values, of length batch_size

    # Convert True or False to ones and zeroes, and calculate the mean
    return correct.float().mean() * 100


def multiple_top_k_accuracy(output, target, K=(1,)):
    """
    Helper function to calculate multiple top-k accuracies. (Optional utility function, not in training loop)
    K is a tuple containg the values k1, k2, ... , etc.
    Outputs are tensors of the format [batch_size, classes=10] (matrix with rows and columns).
    Targets are tensors of the format [batch_size] (a column vector containing true classes).
    """
    results = []
    for k in K:
        results.append(top_k_accuracy(output, target, k))
    return results
    
        
def eval(model, valid_loader, loss_func):
    """
    This method assesses a given model's performance on a testing set.
    Note: The model should be in evaluation mode, so that Batch-Normalization and Dropout layers aren't updated.
    """
    print(f'Evaluating Model on Training Set...')

    losses = AverageMeter()
    top1 = AverageMeter()
    
    model.eval() # Model should be in evaluation mode

    # Disable gradient computation, because Autograd generates gradients in a forward pass
    # This should speed up computation
    with torch.no_grad():
        for _, (inputs, labels) in enumerate(valid_loader):
            # Note that the trainloader consists of [idx, (input, label)] "rows"
            # Make predictions, and calculate loss (forward prop)
            outputs = model(inputs)
            loss = loss_func(outputs, labels)

            # Calculate precision
            prec1 = top_k_accuracy(outputs, labels, k=1)

            # Update perfomance trackers
            # Note that the moving avg should be proportional to the size of the batch
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
    
    print(f'Model Accuracy on Test Set: {top1.avg:.3f}%\n')
    return top1.avg
