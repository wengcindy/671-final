import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
import pandas as pd

"""
Return a list of the aatest losses at the end of each epoch.
"""

def train_test_trajectory(model, optimizer, train_loader, test_loader, criterion, epochs):

    test_trajectory = []
    
    for t in range(epochs):
        
        # TRAIN
        for i, (images, labels) in enumerate(train_loader):
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        

        # TEST
        model.eval()
        test_acc = 0
        total_data = 0
        loss = 0
        with torch.no_grad():
            for _, (images, labels) in enumerate(test_loader):
                output = model(images)
                pred = output.argmax(dim=1, keepdim=True)
                test_acc += pred.eq(labels.view_as(pred)).sum().item()
                total_data += len(images)
                loss = criterion(output, labels)
        print(loss.item())
        test_trajectory.append(loss.item())

    return test_trajectory



"""
For a dataset, will loop through all the optimizers and save the test loss trajectories after training + testing.
"""

def trajectory_loss(train_loader, test_loader, criterion, input_size, num_classes, epochs, batch_size, test_runs):
    
    # Logistic regression model.
    model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(input_size, num_classes),
        torch.nn.LogSoftmax(dim=1) 
    )
    
    # add all optimizers to a list
    optimizer_list=[]
    optimizer_list.append(optim.SGD(model.parameters(), lr=0.01))
    optimizer_list.append(optim.SGD(model.parameters(), lr=0.01,momentum=0.9))
    optimizer_list.append(optim.SGD(model.parameters(), lr=0.01,momentum=0.9,nesterov=True))
    optimizer_list.append(optim.Adagrad(model.parameters(), lr=0.01))
    optimizer_list.append(optim.RMSprop(model.parameters(), lr=0.01))
    optimizer_list.append(optim.Adam(model.parameters(), lr=0.01))
    
    test_losses = [] # store test losses for each optimizer
    
    # carry out training and testing for each optimizer and save the test losses for the number of test runs:
    
    for i in range(test_runs):
        test_run_loss = []
        
        for optimizer in optimizer_list:
            # logistic regression model
            model = torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(input_size, num_classes),
                torch.nn.LogSoftmax(dim=1) 
            )
            trajectory = train_test_trajectory(model, optimizer, train_loader, test_loader, criterion, epochs)

            for i in trajectory:
                test_run_loss.append(i)
      
        test_losses.append(test_run_loss)

    return test_losses



# Use NLL since we include softmax as part of model.  
criterion = nn.NLLLoss()  

all_trajectories = []

# CIFAR-10 dataset

# Normalizer
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

# Data loader
train_loader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

test_loader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)


# Hyperparameters 
input_size = 32*32*3
num_classes = 10


loss_trajectory = trajectory_loss(train_loader, test_loader, criterion, input_size, num_classes, epochs, batch_size, test_runs)
for i in loss_trajectory:
    all_trajectories.append(i)


df = pd.DataFrame(data=all_trajectories)

tasks = ['logistic_regression_gaussian_exact_param']
runs = range(test_runs)
df.index = pd.MultiIndex.from_product([tasks, runs])

optimizers = ['SGD','Momentum','Nesterov','Adagrad','RMSProp','Adam']
epoch_ind = range(epochs)
df.columns = pd.MultiIndex.from_product([optimizers, epoch_ind])

df.to_csv('test')
