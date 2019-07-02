''' Alexandra Rivero 
   :D  '''

import time
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import optim
from collections import OrderedDict
from numpy import transpose
from matplotlib.ticker import FormatStrFormatter
import argparse 
import utility_functions as utilities
import model_functions as md

DEBUG = True

default_data_dir = './flowers'
def init_parser():
    my_argparse = argparse.ArgumentParser()
    my_argparse.add_argument('data_dir', nargs='*', type = str,  help='Path to data_dir', default=default_data_dir)
    my_argparse.add_argument('--data', '--d', default=4, type=int, help='data')
    my_argparse.add_argument('--gpu', action='store_true', help='To activate gpu mode', default=False)
    my_argparse.add_argument('--arch', action='store', type=str, help='Architecture: vgg16, vgg19, vgg13, densenet121 or alexnet', default='vgg16')
    my_argparse.add_argument('--learning_rate', '--lr', default=0.001, type=float, help='To choose an specific learning rate')
    my_argparse.add_argument('--hidden_units', '--hu', default=0, type=int, help='To choose specific hidden units')
    my_argparse.add_argument('--save_dir', '--save', type=str, help='Directory to save the training')

    return my_argparse.parse_args()


def main():
    print("Let's go!")
    custom_args = init_parser()
    print("=>> data_dir '{}'".format(custom_args.data_dir))
    print("=>> --data '{}'".format(custom_args.data))
    device = utilities.get_device(custom_args.gpu)
    print("=>> --gpu '{}'".format(custom_args.gpu))
    # Creatinn the paths 
    train_dir, valid_dir, test_dir = utilities.dirs_initialization(custom_args.data_dir)
    # Loading the datasets with ImageFolder
    train_image_datasets, valid_datasets, test_datasets = utilities.datasets_initialization(train_dir, valid_dir, test_dir )
    # Using the image datasets and the trainforms, define the dataloaders
    train_dataloaders, valid_dataloaders, test_dataloaders =  utilities.dataloaders_initialization(train_image_datasets, valid_datasets, test_datasets)
    if DEBUG:
        print("My model initialization")
    my_model, my_optimizer, my_criterion, my_epochs = md.model_initialization(custom_args.arch, custom_args.learning_rate, custom_args.hidden_units )    
    if DEBUG:
        print("My model training")
        print("=>> --my_model '{}'".format(my_model))
        print("=>> --my_optimizer '{}'".format(my_optimizer))
        print("=>> --train_dataloaders '{}'".format(train_dataloaders))
        print("=>> --valid_dataloaders '{}'".format(valid_dataloaders))
        print("=>> --my_criterion '{}'".format(my_criterion))
        print("=>> --my_epochs '{}'".format(my_epochs))
    
    md.training(my_model, my_optimizer, device, train_dataloaders, valid_dataloaders,  my_criterion, my_epochs)
    if DEBUG:
        print ("End of the training")
    
    if(custom_args.save_dir != None):
        utilities.save_my_checkpoint(my_model, train_image_datasets, custom_args.save_dir, my_optimizer, my_criterion,  custom_args.learning_rate, my_epochs, custom_args.arch)
    
if __name__ == '__main__':
    main()