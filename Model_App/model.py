import torch
from torchvision import models
from config import MODEL_NAME, NUM_CLASS

def create_model(model_name=MODEL_NAME):
    if model_name == 'ResNet':
        # Load the pre-trained model
        model = models.resnet101(weights='DEFAULT')
        
        # Set the output layer to match the number of classes.
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, NUM_CLASS)

    return model