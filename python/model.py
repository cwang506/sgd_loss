from torch.nn import Softplus #smooth relu
import torch.nn as nn
import torch.nn.functional as F
from utils import generate_polynomial_data
import numpy as np
import torch
import torch.optim as optim
from torch.nn import MSELoss
from tqdm import tqdm
import matplotlib.pyplot as plt
# import torch.optim.lr_scheduler.StepLR

class Net(nn.Module):
    def __init__(self, input_size, loss = MSELoss(reduction="sum"), epochs = 3):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(in_features =input_size, out_features = 200)
        self.fc2 = nn.Linear(in_features = 200, out_features = 3000)
        self.fc3 = nn.Linear(in_features = 3000, out_features = 2500)
        self.fc4 = nn.Linear(in_features = 2500, out_features = 4000)
        self.fc5 = nn.Linear(in_features = 4000, out_features = 1)
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
    
    def my_plot(self, epochs, loss):
        plt.plot(epochs, loss)
        plt.show()

    def train_gd(self, data, labels, T, lr):
        optimizer = optim.SGD(self.parameters(), lr = lr)
        n, d = data.shape
        data = torch.from_numpy(data).float()
        loss_list = []
        for epoch in range(self.epochs):
            running_loss = 0.0
            for i in tqdm(range(T)):
                # j = np.random.randint(n)
                # xi = data[j:j+1, :]
                # y = labels[j:j+1, :]
                # data = torch.from_numpy(data).float()
                output = self.forward(data)
                # print(xi, y, output)
                loss = self.loss(output, torch.from_numpy(labels).float())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                # print(loss.item())
                if i%2000 == 1999:
                    print("\rEpoch %s iteration %s loss: %s" %(epoch+1, i+1, round(running_loss/2000, 2)))
            loss_list.append(running_loss/T)
        #print(loss_list)
        self.my_plot(np.linspace(1, self.epochs, self.epochs).astype(int), loss_list)
    
    def train_sgd(self, data, labels, T, lr):
        #need to decay lr
        optimizer_i = optim.SGD(self.parameters(), lr = lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer_i, step_size = 1, gamma = 0.8)
        n, d = data.shape
        data = torch.from_numpy(data).float()
        loss_list = []
        for epoch in tqdm(range(self.epochs)):
            running_loss = 0.0
            for i in range(T):
                if i==0:
                    print(optimizer_i.param_groups[0]['lr'])
                rand_idx = np.random.choice(n)
                data_i = data[rand_idx]
                labels_i = torch.from_numpy(labels[rand_idx])
                output_i = self.forward(data_i)
                loss = self.loss(output_i, labels_i.float())
                running_loss += loss.item()
                optimizer_i.zero_grad()
                loss.backward()
                optimizer_i.step()
                print('\repoch: {}\epochLoss =  {:.3f}'.format(epoch, loss), end="") 
            scheduler.step()
            loss_list.append(running_loss/T)
        #print(loss_list)
        self.my_plot(np.linspace(1, self.epochs, self.epochs).astype(int), loss_list)


def check_loss_landscape(model_state_dict_path, X, Y, sgd = True, loss_function = MSELoss()):
    #check sgd loss landscape, should have every term equal to 0
    n, d = X.shape
    model = Net(d)
    model.load_state_dict(torch.load(model_state_dict_path))
    y_pred = model.forward(torch.from_numpy(X).float())
    
    print("Overall loss: %s"%loss_function(y_pred, torch.from_numpy(Y).float()))
    if sgd:
        for i in range(n):
            datapoint = X[i:i+1, :]
            label = Y[i:i+1, :]
            output = model.forward(torch.from_numpy(datapoint).float())
            print(output.item(), label.item())


    
if __name__ == "__main__":
    torch.manual_seed(0)
    n = 20
    d = 10000
    generate_data = False
    suffix = "3"
    coeffs = np.random.rand(d, 1)
    xvals = np.random.rand(n)
    if generate_data:
        print("Generating Data...")
        X, Y = generate_polynomial_data(coeffs, xvals)
        with open("./datasets/X%s.npy" %suffix, "wb") as f:
            np.save(f, X)
        with open("./datasets/Y%s.npy" %suffix, "wb") as f:
            np.save(f, Y)
        with open("./datasets/coeffs%s.npy" %suffix, "wb") as f:
            np.save(f, coeffs)
    else:
        with open("./datasets/X%s.npy" %suffix, "rb") as f:
            X = np.load(f)
        with open("./datasets/Y%s.npy" %suffix, "rb") as f:
            Y = np.load(f)
        with open("./datasets/coeffs%s.npy" %suffix, "rb") as f:
            coeffs = np.load(f)
    print(X.shape, Y.shape)
    sgd = False
    train = True
    if sgd:
        model_path = "./models/model_sgd_%s.pt"%suffix
        net = Net(d, epochs = 50)
    else:
        model_path = "./models/model_%s.pt"%suffix
        net = Net(d, epochs = 10)
    if train:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            print("Running on GPU")
        else:
            device = torch.device("cpu")
            print("Running on CPU")
        net.to(device)
        if sgd:
            net.train_sgd(X, Y, 100000, lr = 1e-2)
        else:
            net.train_gd(X, Y, 5000, lr = 1e-4)
        torch.save(net.state_dict(), model_path)
    check_loss_landscape(model_path, X, Y, sgd= True)

    
    # torch.save(net.state_dict(), "./models/model")



