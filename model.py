import torch
import torch.nn as nn

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        
        # First Fully Connected layer: Input -> Hidden
        self.l1 = nn.Linear(input_size, hidden_size) 
        
        # Activation function (ReLU)
        # It introduces non-linearity to the network
        self.relu = nn.ReLU()
        
        # Second Fully Connected layer: Hidden -> Hidden
        self.l2 = nn.Linear(hidden_size, hidden_size) 
        
        # Output layer: Hidden -> Output (Classes)
        self.l3 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        """
        Forward pass: defines how data flows through the network.
        """
        out = self.l1(x)
        out = self.relu(out)
        
        out = self.l2(out)
        out = self.relu(out)
        
        out = self.l3(out)
        # no activation and no softmax at the end
        # because CrossEntropyLoss applies softmax internally
        return out
