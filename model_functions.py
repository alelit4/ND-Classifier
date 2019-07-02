import torch
from torchvision import datasets, transforms, models
from collections import OrderedDict
from torch import optim
from torch import nn
from PIL import Image
import matplotlib.pyplot as plt
from numpy import transpose
from matplotlib.ticker import FormatStrFormatter
import utility_functions as utilities
import torch.nn.functional as F
import numpy as np

def model_initialization(type_, learning_rate, hidden_units):
    '''
    Function to define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
    '''
    if type_ == "vgg16":
        model = models.vgg16(pretrained=True)      
    elif type_ == "vgg13":
        model = models.vgg13(pretrained=True)  
    elif type_ == "vgg19":
        model = models.vgg19(pretrained=True)  
    elif type_ == "alexnet":
        model = models.alexnet(pretrained=True)  
    else:
        model = models.densenet121(pretrained=True)  
    
    # Definition of a new untrained feed-forward network
    for param in model.parameters():
        param.requires_grad = False
    
    if hidden_units != 0:
         classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(25088, hidden_units)),
            ('relu1', nn.ReLU()),
            ('fc2', nn.Linear(hidden_units, 102)),
            ('output', nn.LogSoftmax(dim=1)) ]))
    else:
        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(25088, 4096)),
            ('relu1', nn.ReLU()),
            ('fc2', nn.Linear(4096, 2096)),
            ('relu2', nn.ReLU()),
            ('fc3', nn.Linear(2096, 1096)),
            ('relu3', nn.ReLU()),
            ('fc4', nn.Linear(1096, 500)),
            ('relu4', nn.ReLU()),
            ('fc5', nn.Linear(500, 102)),
            ('output', nn.LogSoftmax(dim=1)) ]))
    
    model.classifier = classifier   
    optimizer = optim.Adam(model.classifier.parameters(), learning_rate)
    num_epochs = 10
    criterion = nn.NLLLoss()

    return model, optimizer, criterion, num_epochs

def data_validation(dataloaders, model, device, criterion, optimizer):
    '''
    Function to track the loss and accuracy on the validation set to determine the best hyperparameters
    '''
    accuracy = 0
    loss = 0
    model.to(device)
    for ii, (inputs, labels) in enumerate(dataloaders):
        inputs, labels = inputs.to(device) , labels.to(device)
        outputs = model.forward(inputs)
        loss = criterion(outputs,labels)
        ps = torch.exp(outputs).data
        equality = (labels.data == ps.max(1)[1])
        model.eval()
        accuracy += equality.type_as(torch.FloatTensor()).mean()
    return accuracy, loss   


def training (model, optimizer, device, train_dataloaders, valid_dataloaders, criterion, num_epochs):
    ''' Training the classifier layers using backpropagation using the pre-trained network to get the features.
    '''
    print_every = 20
    steps = 0
    loss_show=[]
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    model.to(device)
    for epoch in range(num_epochs):
        training_loss = 0
        model.train()
        for ii, (inputs, labels) in enumerate(train_dataloaders):
            steps += 1
            inputs,labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            training_loss += loss.item()
            print('.', end='', flush=True)
            if steps % print_every == 0:
                model.eval()
                with torch.no_grad():
                    accuracy, valid_loss = data_validation(valid_dataloaders, model, device, criterion, optimizer)
                    print("Epoch: {}/{}... ".format(epoch+1, num_epochs),
                        "Loss: {:.4f}".format(training_loss/print_every),
                        "Validation Lost {:.4f}".format(valid_loss /len(valid_dataloaders)),
                        "Accuracy: {:.4f}%".format((accuracy /len(valid_dataloaders))*100))
                training_loss = 0
                model.train()

def test_validation(model, test_dataloaders):
    model.eval()
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to('cuda')
    right = 0
    total = 0
    with torch.no_grad():
        for ii in test_dataloaders:
            inputs,labels = ii
            inputs,labels = inputs.to('cuda'), labels.to('cuda')
            outputs = model(inputs)
            
            prediction = torch.max(outputs.data, 1)
            total += labels.size(0)
            eq = (prediction[1] == labels).sum().item()
            right += eq
    print('The final accuracy or the NN based on the test images is: {:.4f}%'.format(100*right/total) )         
    
def predict(image_path, model, topk, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # Implementation of the code to predict the class from an image file
    model.eval()
    model.to(device)
    img_processed = utilities.process_image(image_path).unsqueeze(0)
    img_processed = img_processed.to(device)
    with torch.no_grad():
        outputs = model.forward(img_processed)
        _probs, _classes = outputs.topk(topk)
        _probs, _classes =  _probs.to(device), _classes.to(device)

    probs = F.softmax(_probs, dim=1) 
    classes = np.array(_classes[0])
    class_keys = {x:y for y, x in model.class_to_idx.items()}
    classes = [class_keys[i] for i in classes]
    return np.array(probs.data[0]), classes

