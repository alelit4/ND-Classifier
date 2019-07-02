''' Alexandra Rivero 
   :D  '''

import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import datasets, transforms, models
import json
from collections import OrderedDict
import model_functions as md

def get_data_transforms():
    return transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

def get_valid_transforms():
    return transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

def get_test_transforms():
    return transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

def dirs_initialization(default_data_dir='./flowers'):
    train_dir = default_data_dir + '/train'
    valid_dir = default_data_dir + '/valid'
    test_dir = default_data_dir + '/test'
    return train_dir, valid_dir, test_dir


# Loading the datasets with ImageFolder
def datasets_initialization(train_dir, valid_dir, test_dir):
    train_image_datasets = datasets.ImageFolder(train_dir, transform=get_data_transforms())
    valid_datasets = datasets.ImageFolder(valid_dir, transform=get_valid_transforms())
    test_datasets = datasets.ImageFolder(test_dir, transform=get_test_transforms())
    return  train_image_datasets, valid_datasets, test_datasets

# Using the image datasets and the trainforms, define the dataloaders
def dataloaders_initialization(train_image_datasets, valid_datasets, test_datasets ):
    train_dataloaders = torch.utils.data.DataLoader(train_image_datasets, batch_size=64, shuffle=True)
    valid_dataloaders = torch.utils.data.DataLoader(valid_datasets, batch_size=32)
    test_dataloaders = torch.utils.data.DataLoader(test_datasets, batch_size=32)
    return train_dataloaders, valid_dataloaders, test_dataloaders

def get_device(gpu):
    if gpu:
        return "cuda"
    else:
        return "cpu"

# Save the checkpoint 
def save_my_checkpoint(model, train_image_datasets, save_dir, optimizer, criterion, learning_rate, my_epochs, my_arch):
    model.class_to_idx = train_image_datasets.class_to_idx
    my_checkpoint = {
        'arch' : my_arch,
        'class_to_idx' : model.class_to_idx,
        'input_size' : 25088 ,
        'output_size' : 102,
        'state_dict': model.state_dict(), 
        'epochs' : my_epochs,
        'learning_rate': learning_rate,
        'optimizer_state': optimizer.state_dict(),
        'criterion_state': criterion.state_dict()
    }
    print("Llega a crear el checkpoint")
    torch.save(my_checkpoint, save_dir)
    
# A function that loads a checkpoint and rebuilds the model
def load_my_checkpoint(my_checkpoint_file):
    my_checkpoint = torch.load(my_checkpoint_file)
    my_new_model =  getattr(models, my_checkpoint['arch'])(pretrained = True)
    my_new_model, my_new_optimizer, my_new_criterion, my_epochs = md.model_initialization(my_checkpoint['arch'], my_checkpoint['learning_rate'], 0 )  
    my_new_model.class_to_idx = my_checkpoint['class_to_idx']
    my_new_model.load_state_dict(my_checkpoint['state_dict'])

    return my_new_model, my_new_optimizer, my_new_criterion


def load_cat_to_names(category_names_file):
    with open(category_names_file, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    custom_image = Image.open(image)
    
    custom_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    
    image_transformed = custom_transforms(custom_image)
    return image_transformed

def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy()
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def print_results(path):
    plt.rcParams["figure.figsize"] = (13,13)
    plt.subplot(211)
    
    probs, classes  = predict(path, model)    
    image = process_image(path)

    ax1 = imshow(image, ax = plt)
    ax1.axis('off')
    ax1.title(cat_to_name.get(classes[0]).capitalize())
    ax1.show()
    
    values = []
    for k in classes:
        #print(k, cat_to_name.get(k))
        values.append(cat_to_name.get(k).capitalize())
  
    fig,ax2 = plt.subplots(figsize=(4,4))
    _ticks = [5, 4, 3, 2, 1]
    ax2.barh(_ticks, probs, 0.9, linewidth=5.0, align = 'center')
    ax2.set_yticks(ticks = _ticks)
    ax2.set_yticklabels(values)
    ax2.set_ylim(min(_ticks) - 0.6, max(_ticks) + 0.6)
    
    ax2.set_xticks([0.0,0.2,0.4,0.6,0.8,1,1.2])
    ax2.set_xlim((0,1))
    ax2.xaxis.grid(True)
    ax2.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    plt.show()
