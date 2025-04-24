import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class SurrogateModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SurrogateModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, x):
        return self.layers(x)

class SurrogateTrainer:
    def __init__(self, input_dim, output_dim, lr=0.001):
        self.model = SurrogateModel(input_dim, output_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
    
    def train(self, x_train, y_train, epochs=100):
        x_train, y_train = torch.FloatTensor(x_train), torch.FloatTensor(y_train)
        for _ in range(epochs):
            self.optimizer.zero_grad()
            predictions = self.model(x_train)
            loss = self.criterion(predictions, y_train)
            loss.backward()
            self.optimizer.step()
    
    def predict(self, x):
        with torch.no_grad():
            return self.model(torch.FloatTensor(x)).numpy()
