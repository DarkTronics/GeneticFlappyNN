import torch
import torch.nn as nn
import torch.nn.functional as F

class FlappyBirdNN(nn.Module):
    def __init__(self):
        super(FlappyBirdNN, self).__init__()
        self.fc1 = nn.Linear(3, 16)   # Input layer (3 inputs) to hidden layer
        self.fc2 = nn.Linear(16, 8)   # Hidden layer to another hidden layer
        self.fc3 = nn.Linear(8, 1)    # Hidden layer to output layer (1 output)

    def forward(self, x):
        x = F.relu(self.fc1(x))       # Apply ReLU activation
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # Output probability of flapping
        return x
    
    def crossover(self, other):
        """Create a new child model by randomly combining parameters from self and other."""
        child = FlappyBirdNN()
        child_dict = child.state_dict()
        self_dict = self.state_dict()
        other_dict = other.state_dict()
        for key in child_dict:
            # For each tensor, pick each value from one of the parents at random
            mask = torch.rand_like(self_dict[key]) < 0.5
            child_dict[key] = torch.where(mask, self_dict[key], other_dict[key])
        child.load_state_dict(child_dict)
        return child

    def mutate(self, rate=0.1, mutation_strength=0.5):
        """Mutate the model's parameters by adding random noise at a given mutation rate."""
        for param in self.parameters():
            if len(param.shape) > 0:
                mutation_mask = torch.rand_like(param) < rate
                noise = torch.randn_like(param) * mutation_strength
                param.data.add_(mutation_mask.type(torch.float32) * noise)
