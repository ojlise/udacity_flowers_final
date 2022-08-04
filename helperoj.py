import argparse
import torch
from torch import nn
from torchvision import models

#test

def get_train_input_args():

    # input command line arguments, defaults given if appropriate
    # 
    parser = argparse.ArgumentParser()
    
    parser.add_argument('data_dir', type = str, help = 'path to flower images')    
    parser.add_argument('--save_dir', type = str, default = 'checkpoints', help = 'Save folder for model checkpoints') 
    parser.add_argument('--arch', type = str, default = 'densenet121', choices=['vgg16', 'densenet121'], help = 'Model Architecture') 
    parser.add_argument('-l', '--learning_rate', type = float, default = 0.001, help = 'Learning rate') 
    parser.add_argument('-e', '--epochs', type = int, default = 1, help = 'number of traning Epochs') 
    parser.add_argument('-h1', '--hidden_units', type = int, default = 512, help = 'Hidden units')
    parser.add_argument('-cp', '--checkpoint_path', type = str, help = 'Path to store checkpoint')


    in_args = parser.parse_args()
    
    #print train arguments
    
    if in_args is None:
        print("* Ship input arguments")
    else:
        print("Command Line Arguments:\n dir =", in_args.data_dir, 
              "\n save_dir =", in_args.save_dir, 
              "\n arch =", in_args.arch, 
              "\n learning_rate =", in_args.learning_rate, 
              "\n epochs =", in_args.epochs,
              "\n hidden_units =", in_args.hidden_units)
    
    if in_args.checkpoint_path is not None:
        print("\n checkpoint_path =", in_args.checkpoint_path)
        
    if in_args.gpu is not None:
        print("\n Use gpu if available")

    return in_args




def get_predict_input_args():
    
    # input command line arguments, defaults given if appropriate
    # 
    parser = argparse.ArgumentParser()
    
    parser.add_argument('image_path', type = str, help = 'Path of the prediction image')    
    parser.add_argument('checkpoint_path', type = str, default = 'checkpoints_test/checkpoint_best_model.pth', help = 'Checkpoint path of the trained model') 
    parser.add_argument('-k', '--top_k', type = int, default = 1, help = 'number top most likely classes') 
    parser.add_argument('-json', '--category_names_path', type = str, default = "cat_to_name.json", help = 'JSON file to map categories to flowers')

    in_args = parser.parse_args()
    
    # print predict argumants
    
    if in_args is None:
        print("* Skip input arguments")
    else:
        print("Command Line Arguments:\n image_path =", in_args.image_path, 
              "\n checkpoint_path =", in_args.checkpoint_path, 
              "\n top_k =", in_args.top_k, 
              "\n category_names_path =", in_args.category_names_path)
        
    if in_args.gpu is not None:
        print("\n Use gpu if available")

    return in_args



def build_model(arch, hidden_units, checkpoint=None):
    
    pre_trained_archs = {
        "vgg16": 25088,
        "densenet121": 1024
    }

    model = eval("models.{}(pretrained=True)".format(arch))
    
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
        
    fc1_input = pre_trained_archs[arch]

    from collections import OrderedDict
    classifier = nn.Sequential(OrderedDict([
                              ('dropout1', nn.Dropout(p=0.3)),
                              ('fc1', nn.Linear(fc1_input, hidden_units)),
                              ('relu', nn.ReLU()),
                              ('fc2', nn.Linear(hidden_units, 102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))

    model.classifier = classifier
    
    # if we're using a pretrained model the state_dict will update it to trained status
    if checkpoint is not None:
        model.class_to_idx = checkpoint['class_to_idx']
        model.load_state_dict(checkpoint['state_dict'])

    return model


def test_model(model, testloader): 
# Model in evaluation mode
   # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    model.to(device);
    model.eval()

    #Initilaize accuracy
    accuracy = 0

    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        
        # Class with the highest probability is our predicted class
        equality = (labels.data == outputs.max(1)[1])

        # Accuracy is number of correct predictions divided by all predictions
        accuracy += equality.type_as(torch.FloatTensor()).mean()
        
    print("Test accuracy: {:.3f}".format(accuracy/len(testloader)))
    

    
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    hidden_units = checkpoint['hidden_units']
    arch = checkpoint['arch']
    best_accuracy = checkpoint['best_accuracy']
    #optimizer = checkpoint['optimizer_state_dict']
    class_to_idx = checkpoint['class_to_idx']
    state_dict = checkpoint['state_dict']
    
  #  for param in model.parameters():
   #     param.requires_grad = False
        
    return checkpoint, checkpoint['best_accuracy']