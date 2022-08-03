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

from helper import get_predict_input_args, load_checkpoint, build_model



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





def predict(image_path, model, topk, useGPU=True):
    
    # Use GPU if it's available
    device = torch.device("cuda" if torch.cuda.is_available() and useGPU else "cpu")
    print(f"Device: {device}")
    
    


    #img = Image.open(image_path)
    img = process_image(image_path)
    
    #image = image_path.unsqueeze_(0)
    
    print
    
    # starting prediction    
    model.eval()
    model.to(device);
        
        
        
    image = img.unsqueeze_(0)
    
    with torch.no_grad():
        inputs = image.to(device)
        output = model.forward(inputs)
        probability = F.softmax(output.data,dim=1)
        
        return probability.topk(topk)    
    
    
    
    # Convert 2D image to 1D vector
   # img = np.expand_dims(img, 0)
    #print("np.expand_dims")
        
        
   # img = torch.from_numpy(img)
    #print("torch.from_numpy")

    #print("model.eval()")
    #inputs = Variable(img).to(device)
   # inputs = img.to(device)
    #print("img.to(device)")
    logits = model.forward(inputs)
    #print("model.forward(inputs)")
        
   # ps = F.softmax(logits.data,dim=1)
   # print("F.softmax(logits.data,dim=1)")
  #  topk =ps.cpu().topk(topk)
   # print("ps.cpu().topk(topk)")
    #print("topk",topk)
   
    
    return(topk)
    #return (e.data.numpy().squeeze().tolist() for e in topk)


def main():
    input_args = get_predict_input_args()
    
    #Load checkpoint file (stored model)
    checkpoint, best_accuracy = load_checkpoint(input_args.checkpoint_path)
    #model.eval()
    
    
     # Build model
    model = build_model(checkpoint["arch"],
                         checkpoint["hidden_units"], 
                        checkpoint)

    #print(model)
    useGPU = input_args.gpu is not None
    
    #print("image_path",input_args.image_path,"top_k",input_args.top_k)
    #class_names = list(class_to_idx_dict.keys())
    
    #print("class_names",class_names)
    
    #print("class_name 5",class_names[5])
    

    
    
    # preprocess image
    #preprocessed_image = process_image(input_args.image_path)
    
    # predict image
    #probs, classes = predict(input_args.image_path, model,  input_args.top_k, useGPU)
    top_img = predict(input_args.image_path, model,  input_args.top_k, useGPU)
    
    #print("probs",probs,"classes",classes)
    
    #with open(input_args.category_names_path, 'r') as f:
     #   cat_to_name = json.load(f)
        
    #print("cat 5",cat_to_name[class_names[5]])
    
    #print("cat_to_name",cat_to_name)
    # get top scoring flowers
    #flower_names = [cat_to_name[class_names[e]] for e in classes]
    #flower_names = [cat_to_name[class_to_idx_dict.values()[e]] for e in class_to_idx_dict.keys()]
    
    
    
    # Show result
    with open(input_args.category_names_path, 'r') as f:
        cat_to_name = json.load(f)
        
    
    probs = top_img[0][0].cpu().numpy()
    categories = [cat_to_name[str(category_index+1)] for category_index in top_img[1][0].cpu().numpy()]
    
    for i in range(len(probs)):
        print("TopK {}, Probability: {}, Category: {}\n".format(i+1, probs[i], categories[i]))
    
    
    
if __name__ == '__main__':
    main()