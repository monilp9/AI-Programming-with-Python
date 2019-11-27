import argparse
from collections import OrderedDict
from time import time
import numpy as np
import torch
import json
import torch.nn.functional as F
import torchvision 
from torch import nn, optim
from PIL import Image
from torchvision import datasets, transforms, models

#run by: python predict.py "flowers/test/24/image_06815.jpg" --gpu True


#get input arguments
def get_inputs():
    parser = argparse.ArgumentParser(description='Image classifier.')
    parser.add_argument('input', type=str, help='Required. Please provide the path to the image that should be predicted.')
    #parser.add_argument('checkpoint', type=str, help='checkpoint to load')
    parser.add_argument('--top_k', type=int, default=5, help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='File witch includes the mapping of categories to real names')
    parser.add_argument('--gpu', type=bool, default=False, const=True, nargs='?', help='Set to true if you want to enable and train on GPU.')

    return parser.parse_args()


#main: call the functions:
def main():
    args = get_inputs()  
    
    #loading the checkpoint
    model = models.vgg16(pretrained=True)
    state = torch.load('checkpoint.pth')
          
    model.load_state_dict = state['model_state_dict']
    model.classifier = state['classifier']
    model.arch = state['arch']
    model.optimizer_state_dict = state['optimizer_state_dict']
    model.class_to_idx = state['class_to_idx']
    model.epochs = state['epochs'] 
    model.input_size = state['input_size']
    model.output_size = state['output_size']
    model.hidden_layers = state['hidden_layers']
    model.learning_rate = state['learning_rate']     
          
         
    predict(args.input, model, args.top_k)
          
#This script predicts a label based on the model and image given as an argument when running the script.
def predict(image_path, model, topk=5):
    if torch.cuda.is_available():
        model.cuda()
    else:
        model.cpu()
        
    prediction_img = process_image(image_path) #we processed it 
    predicted_img = prediction_img.unsqueeze(0).cuda() 
    
    output = model(predicted_img.float())
    print(output)
    # probs, indices = torch.max(output.data,1)
    probs, indices = torch.topk(F.softmax(output, dim=1), topk, sorted=True)
    print("probs: ", probs, "\n indices: ", indices)
    ps = F.softmax(output.data,dim=1)

    probs = np.array(ps.topk(topk)[0][0])
    # print('length: ', len(probs))
    index_to_class = {}
    num = 0
    for key, val in model.class_to_idx.items():
        num = num + 1
        index_to_class[val]= key

    class_names = []
    print(ps.topk(topk)[0][0])
    for i in np.array(ps.topk(topk)[0][0]):

        ps_list = ps[0].tolist()
        index_val = ps_list.index(i)
        class_names.append(np.int_(index_to_class[index_val]))


    classes = []

    for i in np.array(class_names):
        classes.append(cat_to_name[str(i)])
    
    for i in range(0,5):
        print("Model predicted: {} flower with a probablity of {}%".format(classes[i], 100*probs[i]))
          
    return probs,class_names
          
           
if __name__ == "__main__":
    main()
