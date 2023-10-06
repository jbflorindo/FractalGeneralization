import numpy as np
import datetime
now = datetime.datetime.now

import torch
from torch import nn, optim
import torchvision
from torchvision import transforms
import copy
import pdb
#%% Train the model for one epoch on the given set
def train(model, device, train_loader, criterion, optimizer, epoch):
    sum_loss, sum_correct = 0, 0

    # Switch to train mode
    model.train()

    for i, (data, target) in enumerate(train_loader):
        data, target = data.to(device, dtype=torch.float), target.to(device)

        # Compute the output
        output = model(data)

        # Compute the classification loss and accuracy
        #pdb.set_trace()
        loss = criterion(output, target)
        pred = output.max(1)[1]
        sum_correct += pred.eq(target).sum().item()
        sum_loss += len(data) * loss.item()

        # Compute the gradient and do an SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Return the error and cross-entropy loss
    return 1 - (sum_correct / len(train_loader.dataset)), sum_loss / len(train_loader.dataset)


# Evaluate the model on the given set
def validate(model, device, val_loader, criterion):
    sum_loss, sum_correct = 0, 0
    
    # Switch to evaluation mode
    model.eval()
        
    with torch.no_grad():
        for i, (data, target) in enumerate(val_loader):
            data, target = data.to(device, dtype=torch.float), target.to(device)

            # Compute the output
            output = model(data)

            # Compute the classification loss and accuracy
            loss = criterion(output, target)
            pred = output.max(1)[1]
            sum_correct += pred.eq(target).sum().item()
            sum_loss += len(data) * loss.item()

    # Return error and cross-entropy loss
    return 1 - (sum_correct / len(val_loader.dataset)), sum_loss / len(val_loader.dataset)

#%% Define training parameters 

# Images channels and number of classes
nchannels, nclasses  = 3, 10

# Other parameters
batch_size = 64#32,64

# Total number of epochs
epochs = 2000 # 2000
# If verbose=1 print the training and validation loss and accuracy
verbose = 1 #0

# Device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Import the model
from codes.model import Network_SingleTask

#%% Varying hyperparameters
#learningrate_values = [0.00032,0.0001,0.000032]
learningrate_values = [0.000032]
nbrblocks_values = [2,3,4,5,6,7]
#nbrblocks_values = [5,6,7]

# Tensors for losses and errors
train_error = torch.zeros(1, dtype=torch.float64, device=device)
train_loss = torch.zeros(1, dtype=torch.float64, device=device)
test_error = torch.zeros(1, dtype=torch.float64, device=device)
test_loss = torch.zeros(1, dtype=torch.float64, device=device)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#==============================================================================
"""# CIFAR 10
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=0)"""

#==============================================================================
# SVHN
from torch.utils.data import random_split
dataset = torchvision.datasets.SVHN(root='data/', download=True, transform=transforms.ToTensor())
val_size = 12000
train_size = len(dataset) - val_size

train_ds, val_ds = random_split(dataset, [train_size, val_size])

trainloader = torch.utils.data.DataLoader(train_ds, batch_size, shuffle=True, num_workers=0, pin_memory=True)
testloader = torch.utils.data.DataLoader(val_ds, batch_size*2, num_workers=0, pin_memory=True)
#==============================================================================

for learningrate in learningrate_values:
    
    for nbr_blocks in nbrblocks_values:

        # Keep track of cross-entropy losses and errors
        train_losses = []
        valid_losses = []
        train_errors = []
        valid_errors = []
    
        # Create an instance of the model
        # Disable regularization via batch normalization and dropout
        model = Network_SingleTask(nchannels, nclasses, batch_norm=False, dropout=False, nbr_blocks=nbr_blocks)
        #model = NiN(nclasses)
        model = model.to(device)
    
        # Create a copy of the initial model, will be used for calculating complexity measures
        init_model = copy.deepcopy(model)
    
        # Define loss function (criterion) and optimizer
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = optim.Adam(model.parameters(), lr=learningrate)
    
        # Record the time (to report the training time)
        t = now()
    
        weights_list = []
        # Training the model
        for epoch in range(0, epochs):
    
            # Train for one epoch
            tr_err, tr_loss = train(model, device, trainloader, criterion, optimizer, epoch)
            
            #weights_list.append(get_vec_params(model).detach().cpu().numpy())        
    
            # Evaluate after each epoch
            val_err, val_loss = validate(model, device, testloader, criterion)
    
            train_losses.append(tr_loss)
            valid_losses.append(val_loss)
            train_errors.append(tr_err)
            valid_errors.append(val_err)
    
            # Display after each epoch
            if verbose==1:
                print(f'Epoch: {epoch + 1}/{epochs}\t Training loss: {tr_loss:.3f}\t',
                      f'Training error: {tr_err:.3f}\t Validation error: {val_err:.3f}')
    
            # Stopping condition: training loss < 0.01
            if tr_loss <= 0.01: 
                print('Stopping.' )
                early_stop = True
                
                train_error = train_errors[-1]
                train_loss = train_losses[-1]
                train_acc = 1-train_error
            
                # Calculate the test error and cross-entropy loss of the learned model
                test_error, test_loss = validate(model, device, testloader, criterion)
                print(f'\nTest dataset performance: Test cross-entropy loss: {test_loss:.3f} \t Test accuracy: {1-test_error:.3f}\n')
                
                np.save('train_error_lr_'+str(learningrate)+'_blocks_'+str(nbr_blocks)+'_batch_'+str(batch_size)+'.npy', train_error)
                np.save('test_error_lr_'+str(learningrate)+'_blocks_'+str(nbr_blocks)+'_batch_'+str(batch_size)+'.npy', test_error)
                
                torch.save(model,'model_lr_'+str(learningrate)+'_blocks_'+str(nbr_blocks)+'_batch_'+str(batch_size)+'.pth')
                torch.save(trainloader,'trainloader_lr_'+str(learningrate)+'_blocks_'+str(nbr_blocks)+'_batch_'+str(batch_size)+'.pth')
                
                break
            else:
                continue
        
        print('Training time: %s' % (now() - t))