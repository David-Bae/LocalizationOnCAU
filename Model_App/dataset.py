from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from config import *


transform = transforms.Compose([transforms.Resize((RESIZE_TO, RESIZE_TO))
                                ,transforms.ToTensor()
                                #,transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                ])

def create_dataset(data_path):
    dataset = torchvision.datasets.ImageFolder(root = data_path,
                                           transform = transform)
    return dataset
    
def create_dataloader(trainset, batch_size=BATCH_SIZE):
    dataloader = DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=4)
    return dataloader

if __name__ == '__main__':
    trainset = create_dataset(TRAIN_FOLDER_PATH)
    class_names = trainset.classes
    print(class_names)