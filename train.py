# creates a predciton of flowers from cobnination of a transfer network and classifier.

# to run using terminal

# python train.py <flowr image directory> --save_dir <checkpoint directory> -l <learning rate> -e <training epocs> -h1 <hidden layer>

# Example: 

# python python train.py flowers --save_dir checkpoints_test -l 0.001 -e 1 

import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from PIL import Image
import seaborn as sns
import numpy as np

import time
from torch.autograd import Variable

#import helper 
from helperoj import get_train_input_args, load_checkpoint, build_model, test_model #, save_checkpoint


# save state of trained model
def save_checkpoint(model, train_data, path, best_accuracy, arch, hidden_units):
    model.class_to_idx = train_data.class_to_idx
    torch.save({'state_dict': model.state_dict(),
                'class_to_idx': model.class_to_idx,
                'best_accuracy': best_accuracy,
                'arch': arch,
               'hidden_units': hidden_units
               },
                path)

def main():
        
    input_args = get_train_input_args()
    
    # Create & adjust data
    train_dir = input_args.data_dir + '/train'
    valid_dir = input_args.data_dir + '/valid'
    test_dir = input_args.data_dir + '/test'
    
    print("\n\n Trainings folder: {}".format(train_dir))
    print(" Validation folder: {}".format(valid_dir))
    print(" Test folder: {}\n".format(test_dir))
    
    
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    # Pass transforms in here, then run the next cell to see how the transforms look
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64, shuffle=True)
    
    
    # Load checkpoint
    checkpoint = None
    best_accuracy = 0
    
    if input_args.checkpoint_path is not None:
        checkpoint, best_accuracy = load_checkpoint(input_args.checkpoint_path)
        
    useGPU = input_args.gpu is not None
    
    arch = input_args.arch if checkpoint is None else checkpoint["arch"]
    
    hidden_units = input_args.hidden_units if checkpoint is None else checkpoint["hidden_units"]
    
    # Build model
    model = build_model(arch,
                        hidden_units, 
                        #hidden_units_02, 
                        checkpoint)

    
    # Train model
    print("\n\nStart Training...\n")
    
    
        
    # initialize model    
    epochs = input_args.epochs
    learning_rate = input_args.learning_rate
    
    steps = 0
    running_loss = 0
    print_every = 5
    
    #Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    #Use GPU if it's available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model.to(device);

    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)
            
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        
                        test_loss += batch_loss.item()
                        
                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                        
                print(f"Epoch {epoch+1}/{epochs}.. "
                    f"Train loss: {running_loss/print_every:.3f}.. "
                    f"Test loss: {test_loss/len(testloader):.3f}.. "
                    f"Test accuracy: {accuracy/len(testloader):.3f}")
                
                if best_accuracy < accuracy/len(validloader) and accuracy/len(validloader) > 0.6:
                    best_accuracy = accuracy/len(validloader)
                    path = input_args.save_dir + "/checkpoint_best_model"      
                    print(path)

                    save_checkpoint(model, train_data, path, best_accuracy, arch, hidden_units)

                
                running_loss = 0
                
     # Test trained model
    test_model(model, testloader)
    
    # save whole model
    #torch.save(model, 'checkpoints_test/checkpointWholeModel.pth')
    
if __name__ == '__main__':
    main()