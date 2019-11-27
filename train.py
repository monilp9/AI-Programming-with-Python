#Printtraining loss
#Print validation loss
#Print validation accuracy as the network trains

# Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
# Choose architecture: python train.py data_dir --arch "vgg13"
# Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
# Use GPU for training: python train.py data_dir --gpu



#How to run it? : python train.py "flowers" --arch "vgg16" --gpu True


#import modules 
import argparse
from collections import OrderedDict
from time import time
import numpy as np
import torch
import os
import json
import torch.nn.functional as F
import torchvision 
from torch import nn, optim
from PIL import Image
from torchvision import datasets, transforms, models



#load the data
def load_data(data_dir):
    
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    for i in [train_dir, valid_dir, test_dir]:
        if not os.path.isdir(i):
            print("Directory " + i + " does not exist. Please check the path or the name of the directory.")
                  
    data_transforms = {
    'train_transforms' : transforms.Compose([transforms.RandomRotation(50),
                         transforms.RandomResizedCrop(224), transforms.ToTensor(), 
                         transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])]),
    
    'validation_transforms' : transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                              transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])]),
    'test_transforms' : transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                              transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
                }
        #Load the datasets with ImageFolder
    image_datasets = {
        'train_data' : datasets.ImageFolder(train_dir, transform=data_transforms['train_transforms']),
        'validation_data' : datasets.ImageFolder(valid_dir, transform=data_transforms['validation_transforms']),
        'test_data' : datasets.ImageFolder(test_dir ,transform = data_transforms['test_transforms'])
                    }
        # TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {
        'train_loader' : torch.utils.data.DataLoader(image_datasets['train_data'], batch_size=64, shuffle=True),
        'validation_loader' : torch.utils.data.DataLoader(image_datasets['validation_data'], batch_size=64, shuffle=True),
        'test_loader' : torch.utils.data.DataLoader(image_datasets['test_data'], batch_size=64, shuffle=True)    
                    }
    return data_transforms, image_datasets, dataloaders


def load_arch(arch):

    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        input_size = 25088
    elif arch == 'alexnet':
        model = models.alexnet(pretrained=True)
        input_size = 9216
    elif arch == 'resnet18':
        model = models.resnet18(pretrained=True)
        input_size = 512
    elif arch == 'densenet121':
        model = models.densenet121(pretrained=True)
        input_size = 1024
    else:
        print('Please choose one of \'vgg16\', \'alexnet\', \'resnet18\' or , \'densenet121\' for arch argument.')
        
    for param in model.parameters():
        param.requires_grad = True
        
    return model, input_size

    
def validation(model, testloader, criterion):
    test_loss = 0
    accuracy = 0
    for x, (images2, labels2) in enumerate(testloader):
        if torch.cuda.is_available():
            model.cuda()
            images2 = images2.to('cuda')
            labels2 = labels2.to('cuda')
            
        #output = model(images2)
        output = model.forward(images2)
        test_loss += criterion(output, labels2).item()

        ps = torch.exp(output).data
        equality = (labels2.data == ps.max(1)[1])
        accuracy += equality.type(torch.cuda.FloatTensor).mean()
    
    return test_loss, accuracy

def train_model(model, input_size, data_transforms, image_datasets, dataloaders):
     
    input_size = input_size        
    hidden_sizes = [120,90]  
    output_size = 102       

    for parameter in model.parameters():
        parameter.requires_grad = True 
    
    classifier = torch.nn.Sequential(OrderedDict([
                      ('fc1', nn.Linear(model.classifier[0].in_features, hidden_sizes[0])),
                      # ('fc1', nn.Linear(25588, hidden_sizes[0])),
                      ('relu1', nn.ReLU()),
                      ('dropout', nn.Dropout(p=0.15)),
                      ('fc2', nn.Linear(hidden_sizes[0], hidden_sizes[1])),
                      ('relu2', nn.ReLU()),
                      ('dropout', nn.Dropout(p=0.15)),
                      ('output', nn.Linear(hidden_sizes[1], output_size)),
                      ('softmax', nn.LogSoftmax(dim=1))]))

    model.classifier = classifier
    
    
    epochs = 2
    steps = 0
    learning_rate = 0.05
    #running_loss = 0
    criterion = nn.NLLLoss()    ## you could try nn.NLLLoss()           
    optimizer = optim.SGD(model.classifier.parameters(), lr= learning_rate)
    model.to('cuda')
    print_every = 15
    for e in range(epochs):
    
        model.train()
        running_loss = 0
        for images, labels in iter(dataloaders['train_loader']):
        # test code below
            images = images.float()
        # test code above
            steps += 1
            optimizer.zero_grad()
        
    
        
            if torch.cuda.is_available():
                model.cuda()
                images = images.to('cuda')
                labels = labels.to('cuda')
            else:
                model.cpu()
            
        #output = model(images)
            output = model.forward(images)  #cam commented this line out, alternatively try output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
        
            if steps % print_every == 0:
                val_loss = 0
                accuracy = 0
                # Make sure network is in eval mode for inference
                model.eval()
            
            # Turn off gradients for validation, saves memory and computations
                with torch.no_grad():
                    val_loss, accuracy = validation(model, dataloaders['validation_loader'], criterion)
                #output = model(images)   --cam had it in the code, uncomment 
                # output = model.forward(images)           

                
                print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                  "Validation Loss: {:.3f}.. ".format(val_loss/len(dataloaders['validation_loader'])),
                  "Validation Accuracy: {:.3f}".format(accuracy/len(dataloaders['validation_loader'])))
            
                running_loss = 0
    
  
##get input arguments and parse the arguments 
def get_inputs():
    parser = argparse.ArgumentParser(description='Image classifier.')
    
    parser.add_argument('data_dir', type=str, help='Path to the image files. Subdirectories are ./train, ./valid, ./test')
    parser.add_argument('--model_checkpoint', type=str, default='.', help='This is a required argument. Please provide path to the checkpoint.')
    parser.add_argument('--arch', type=str, default='vgg16', help='CNN model architecture to use for image classification. Pick any of the following: vgg16, alexnet, resnet18, densenet121')
    parser.add_argument('--learning_rate', type=float, default=0.05, help='Learning rate for the VGG16 model.')
    parser.add_argument('--hidden_units', type=str, default=[2048, 1000], help='Sizes for hidden layers. If there are more than one, then separate with comma.')
    parser.add_argument('--output_units', type=int, default=102, help='Output size of the network')
    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs.')
    parser.add_argument('--gpu', type=bool, default=False, const=True, nargs='?', help='Enables GPU capabilities.')
    return parser.parse_args()

# def save_checkpoint(model, image_datasets):
#     model.class_to_idx = image_datasets['train_data'].class_to_idx
#     classifier = model.classifier

#     checkpoint = {
#               'arch' : 'vgg16',
#               'classifier' : classifier,
#               'learning_rate' : 0.05,
#               'input_size': 25588,
#               'output_size': 102,
#               'hidden_layers': [2048, 1000],
#               'class_to_idx' : model.class_to_idx,
#               'model_state_dict' : model.state_dict(),
#               'optimizer_state_dict' : optimizer.state_dict(),
#               'epochs' : epochs,
#               }
#     checkpoint_path = 'checkpoint.pth'
#     torch.save(checkpoint, checkpoint_path)

def test_model(model, dataloaders):
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for x, (images, labels) in enumerate(dataloaders['test_loader']):
        # test code below
            images = images.float()
        # print(images)
        # test code above
            if torch.cuda.is_available():
                model.cuda()
                images = images.to('cuda')
                labels = labels.to('cuda')
            #print(labels)
            else:
                model.cpu()
        # output = model.forward(images)
            output = model(images)
        
            probs, probs_label = torch.max(output.data,1)
            total += labels.size(0)
            correct += (probs_label == labels).sum().item()
    print("Accuracy: {}".format(100 * correct/total))

def main():
    args = get_inputs()
    model, input_size = load_arch(args.arch)
    data_transforms, image_datasets, dataloaders = load_data(args.data_dir)
    train_model(model, input_size, data_transforms, image_datasets, dataloaders)
    test_model(model, dataloaders)
    optimizer = optim.SGD(model.classifier.parameters(), lr= args.learning_rate)
    model.class_to_idx = image_datasets['train_data'].class_to_idx
    classifier = model.classifier

    checkpoint = {
              'arch' : 'vgg16',
              'classifier' : classifier,
              'learning_rate' : 0.05,
              'input_size': 25588,
              'output_size': 102,
              'hidden_layers': [2048, 1000],
              'class_to_idx' : model.class_to_idx,
              'model_state_dict' : model.state_dict(),
              'optimizer_state_dict' : optimizer.state_dict(),
              'epochs' : args.epochs,
    }
    torch.save(model, 'checkpoint.pth')
          
          
if __name__ == "__main__":
    main()
