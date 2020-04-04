# torch imports
import torch.nn.functional as F
import torch.nn as nn


## TODO: Complete this classifier
class MultiClassClassifier(nn.Module):
    """
    Define a neural network that performs binary classification.
    The network should accept your number of features as input, and produce 
    a single sigmoid value, that can be rounded to a label: 0 or 1, as output.
    
    Notes on training:
    To train a binary classifier in PyTorch, use BCELoss.
    BCELoss is binary cross entropy loss, documentation: https://pytorch.org/docs/stable/nn.html#torch.nn.BCELoss
    """

    ## TODO: Define the init function, the input params are required (for loading code in train.py to work)
    def __init__(self, input_features, hidden_dim, output_dim):
        """
        Initialize the model by setting up linear layers.
        Use the input parameters to help define the layers of your model.
        :param input_features: the number of input features in your training/test data
        :param hidden_dim: helps define the number of nodes in the hidden layer(s)
        :param output_dim: the number of outputs you want to produce
        """
        super(MultiClassClassifier, self).__init__()

        # define any initial layers, here
        # 2 layers Fully connected Neural Network
        self.fc1 = nn.Linear(input_features, hidden_dim)
        #self.fc2 = nn.Linear(hidden_dim, hidden_dim//2)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        # ReLu layer for non linear activation
        self.relu = nn.ReLU()
        # Dropout layer:
        self.drop = nn.Dropout(0.3)
        # Batchnormalization layers
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        #self.bn2 = nn.BatchNorm1d(hidden_dim//2)

        #self.sig = nn.Sigmoid()
        

    
    ## TODO: Define the feedforward behavior of the network
    def forward(self, x):
        """
        Perform a forward pass of our model on input features, x.
        :param x: A batch of input features of size (batch_size, input_features)
        :return: A single, sigmoid-activated value as output
        """
        
        # define the feedforward behavior
        # Apply a FC layer, BatchNorm and RELU activation on layer 1
        out = self.fc1(x)
        out = self.bn1(out)
        out= self.relu(out)
        # Dropout inputs to layer 2
        out = self.drop(out)
        # Apply a FC layer, BatchNorm and RELU activation on layer 2
        #out = self.fc2(out)
        #out = self.bn2(out)
        #out= self.relu(out)
        # Dropout inputs to layer 3
        #out = self.drop(out)
        # Apply a FC layer on layer 2 (Softmax applied in loss calculation)
        out = self.fc3(out)

        return out
    