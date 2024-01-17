import torch
import torch.nn as nn
import torch.optim as optim
import os
from model import create_model
from dataset import *
from config import *

if __name__ == '__main__':
    if not os.path.exists('./output'):
        os.makedirs('./output')

    model = create_model()
    model = model.to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    trainset = create_dataset(TRAIN_FOLDER_PATH)
    trainloader = create_dataloader(trainset)

    validset = create_dataset(VALID_FOLDER_PATH)
    validloader = create_dataloader(validset)

    testset = create_dataset(TEST_FOLDER_PATH)
    testloader = create_dataloader(testset)

    best_val_loss = float('inf')

    for epoch in range(NUM_EPOCHS):   
        print(f"\nEPOCH {epoch+1} of {NUM_EPOCHS}")
        running_loss = 0.0

        # Training Code
        for i, data in enumerate(trainloader, 0):
            print(f"Iter {i+1} of {len(trainloader)}")
            inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Validation Code
        model.eval()  
        valid_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():  # Disable Gradient Calculation
            for data in validloader:  
                inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        valid_loss /= len(validloader)
        accuracy = 100 * correct / total
        print(f'Validation Loss: {valid_loss:.3f}, Accuracy: {accuracy:.2f}%')

        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            torch.save(model.state_dict(), PRETRAINED_MODEL)
            print(f'Epoch {epoch+1}: Best model saved with val_loss: {valid_loss:.4f}')

        model.train()  

    print('Finished Training')
    print('Testing...')
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  
        for data in testloader:  
            inputs, labels = data[0].to(DEVICE), data[1].to(DEVICE)
            outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_loss /= len(testloader)
    accuracy = 100 * correct / total

    print(f'Test Loss: {test_loss:.3f}, Accuracy: {accuracy:.2f}%')