# Using trained model to predict flower

# process image is called from predict image

# recall the saved checkpoint and rebuild the model before making prediction

# Command line: python predictoj.py 
# python predict.py <path to test image> <path to stored checkpoint> --top_k <top k classes> 

# Example:
# python predictoj.py flowers/test/1/image_06764.jpg checkpoints_test/checkpoint_best_121_3.pth --top_k 3

import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from PIL import Image
import seaborn as sns
import numpy as np
import json

import time
from torch.autograd import Variable

from helperoj import get_predict_input_args, load_checkpoint, build_model



def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    current_img = Image.open(image)
    
    # TODO: Process a PIL image for use in a PyTorch model
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    image = preprocess(current_img)
    return image





def predict(image, model, topk, useGPU=True):
    
    # Use GPU if it's available
    device = torch.device("cuda" if torch.cuda.is_available() and useGPU else "cpu")
   
    # print(f"Device: {device}")
  
    

    #process Image
    img = process_image(image)
    

    # starting prediction    
    model.eval()
    model.to(device);
        
    
    # Convert 2D image to 1D vector
    img = np.expand_dims(img, 0)

        
    img = torch.from_numpy(img)

    #inputs = Variable(img).to(device)
    inputs = img.to(device)
    logits = model.forward(inputs)
    ps = F.softmax(logits.data,dim=1)
   # topk =ps.cpu().topk(topk)
    topk =ps.topk(topk)
    print("topk",topk)
   
    
    return(topk)
    #return (e.data.numpy().squeeze().tolist() for e in topk)


def main():
    input_args = get_predict_input_args()
    
    #Load checkpoint file (stored model)
    checkpoint, best_accuracy = load_checkpoint(input_args.checkpoint_path)
    
    
    # Build model
    model = build_model(checkpoint["arch"],
                        checkpoint["hidden_units"], 
                      checkpoint)
    
    model.load_state_dict(checkpoint['state_dict'])
    

    
    useGPU = input_args.gpu is not None
    

    #probs, classes = predict(input_args.image_path, model,  input_args.top_k, useGPU)
    top_img = predict(input_args.image_path, model,  input_args.top_k, useGPU)
    

    
     #Show result
    with open(input_args.category_names_path, 'r') as f:
        cat_to_name = json.load(f)
        
    #print("input_args.category_names_path",input_args.category_names_path)
    #print("cat_to_name",cat_to_name)
    
    probs = top_img[0][0].cpu().numpy()
    print("probs",probs)
    categories = [cat_to_name[str(category_index+1)] for category_index in top_img[1][0].cpu().numpy()]
    
    #print("categories",categories)
    
    for i in range(len(probs)):
        print("TopK {}, Probability: {}, Category: {}\n".format(i+1, probs[i], categories[i]))
    
    
    
if __name__ == '__main__':
    main()