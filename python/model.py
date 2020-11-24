from torch.nn import Softplus #smooth relu
import torch.nn as nn
import torch.nn.functional as F
from utils import generate_polynomial_data
import numpy as np
import torch
import torch.optim as optim
from torch.nn import MSELoss
from tqdm import tqdm

class Net(nn.Module):
    def __init__(self, input_size, loss = MSELoss(reduction="sum"), epochs = 3):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(in_features =input_size, out_features = 20)
        self.fc2 = nn.Linear(in_features = 20, out_features = 300)
        self.fc3 = nn.Linear(in_features = 300, out_features = 250)
        self.fc4 = nn.Linear(in_features = 250, out_features = 400)
        self.fc5 = nn.Linear(in_features = 400, out_features = 1)
        self.loss = loss
        self.epochs = epochs

    def forward(self, x):
        x = self.fc1(x)
        x = F.softplus(x)
        x = self.fc2(x)
        x = F.softplus(x)
        x = self.fc3(x)
        x = F.softplus(x)
        x = self.fc4(x)
        x = F.softplus(x)
        x = self.fc5(x) #output layer
        return x

    def train(self, data, labels, T):
        optimizer = optim.SGD(self.parameters(), lr = 1e-6)
        n, d = data.shape
        for epoch in range(self.epochs):
            running_loss = 0.0
            for i in tqdm(range(T)):
                optimizer.zero_grad()
                j = np.random.randint(n)
                xi = data[j:j+1, :]
                y = labels[j:j+1, :]
                xi = torch.from_numpy(xi).float()
                output = self.forward(xi)
                # print(xi, y, output)
                loss = self.loss(output, torch.from_numpy(y).float())
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                # print(loss.item())
                if i%2000 == 1999:
                    print("Epoch %s iteration %s loss: %s" %(epoch+1, i+1, round(running_loss/2000, 2)))
    
if __name__ == "__main__":
    n = 5000
    d = 10000
    X, Y = generate_polynomial_data(np.random.rand(d, 1), np.random.rand(n))
    print(X.shape, Y.shape)
    net = Net(d)
    net.train(X, Y,5000)



