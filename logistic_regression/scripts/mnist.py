import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
import pandas as pd



"""
Return a list of the test losses at the end of each epoch.
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
    
    test_losses = [] # store test losses for each optimizer
    
    # carry out training and testing for each optimizer and save the test losses for the number of test runs:
    
    for i in range(test_runs):
        test_run_loss = []
        
        # SGD       
        model = torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(input_size, num_classes),
                torch.nn.LogSoftmax(dim=1) 
        )
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        trajectory = train_test_trajectory(model, optimizer, train_loader, test_loader, criterion, epochs)
        for i in trajectory:
            test_run_loss.append(i)
        
        # Momentum        
        model = torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(input_size, num_classes),
                torch.nn.LogSoftmax(dim=1) 
        )
        optimizer = optim.SGD(model.parameters(), lr=0.01,momentum=0.9)
        trajectory = train_test_trajectory(model, optimizer, train_loader, test_loader, criterion, epochs)
        for i in trajectory:
            test_run_loss.append(i)
        
        # Adadelta        
        model = torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(input_size, num_classes),
                torch.nn.LogSoftmax(dim=1) 
        )
        optimizer = optim.Adadelta(model.parameters(), lr=1.0)
        trajectory = train_test_trajectory(model, optimizer, train_loader, test_loader, criterion, epochs)
        for i in trajectory:
            test_run_loss.append(i)
        
        # Adagrad        
        model = torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(input_size, num_classes),
                torch.nn.LogSoftmax(dim=1) 
        )
        optimizer = optim.Adagrad(model.parameters(), lr=0.01)
        trajectory = train_test_trajectory(model, optimizer, train_loader, test_loader, criterion, epochs)
        for i in trajectory:
            test_run_loss.append(i)
        
        # RMSprop       
        model = torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(input_size, num_classes),
                torch.nn.LogSoftmax(dim=1) 
        )
        optimizer = optim.RMSprop(model.parameters(), lr=0.01)
        trajectory = train_test_trajectory(model, optimizer, train_loader, test_loader, criterion, epochs)
        for i in trajectory:
            test_run_loss.append(i)
        
        # Adam
        model = torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(input_size, num_classes),
                torch.nn.LogSoftmax(dim=1) 
        )
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        trajectory = train_test_trajectory(model, optimizer, train_loader, test_loader, criterion, epochs)
        for i in trajectory:
            test_run_loss.append(i)
   
            
        test_losses.append(test_run_loss)         

    return test_losses



# Use NLL since we include softmax as part of model.  
criterion = nn.NLLLoss()  

all_trajectories = []

# MNIST dataset (images and labels)
train_dataset = torchvision.datasets.MNIST(root='../../data', 
                                           train=True, 
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='../../data', 
                                          train=False, 
                                          transform=transforms.ToTensor())

# Data loader (input pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)

epochs = 50
batch_size = 100
test_runs = 10

# Hyperparameters 
input_size = 28*28
num_classes = 10


loss_trajectory = trajectory_loss(train_loader, test_loader, criterion, input_size, num_classes, epochs, batch_size, test_runs)
for i in loss_trajectory:
    all_trajectories.append(i)


df = pd.DataFrame(data=all_trajectories)

tasks = ['mnist']
runs = range(test_runs)
df.index = pd.MultiIndex.from_product([tasks, runs])

optimizers = ['GradientDescentOptimizer','MomentumOptimizer','AdadeltaOptimizer','AdagradOptimizer','RMSPropOptimizer','AdamOptimizer']
epoch_ind = range(epochs)
df.columns = pd.MultiIndex.from_product([optimizers, epoch_ind])

df.to_csv('mnist.csv')
