import random
from datetime import datetime
from sklearn.preprocessing import StandardScaler, normalize
from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.optim import Adam, lr_scheduler

from dataset import ImageClassificationDataset, data_utils
from yolov5.model import ResNet152, ResNet18, ResNet34


def compute_accuracy_and_loss(model, ds, data_loader, device, loss_fn):
    
    correct = 0
    data_loss = 0
    l = len(ds)
    model.eval()
    with torch.no_grad():
        for images, labels in data_loader:

            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            data_loss += loss_fn(outputs, labels).item()        
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum()

    return float(correct) / l * 100, data_loss


train_acc_lst, test_acc_lst = [], []
train_loss_lst, tset_loss_lst = [], []
max_train_acc = [0]*2
max_test_acc = [0]*2


def train(device, model, epochs, train_ds, train_loader, val_ds, val_loader, loss_fn, optimizer, scheduler):
    
    best_accuracy = 0
    print("The model will be running on", device, "device")
    # Convert model parameters and buffers to CPU or Cuda
    model.to(device)
    model.train()
    for epoch in range(epochs):  # loop over the dataset multiple times
        
        tStart = datetime.now()
        pbar = tqdm(train_loader)

#-----------------------------------------------------------------------------------------------
        for i, (images, labels) in enumerate(pbar):
            
            # get the inputs
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
#-----------------------------------------------------------------------------------------------
        scheduler.step()

        with torch.no_grad():

            train_accuracy, train_loss = compute_accuracy_and_loss(model, train_ds, train_loader, device, loss_fn)

            if train_accuracy > max_train_acc[1]:
                max_train_acc[0] = epoch+1
                max_train_acc[1] = train_accuracy
            
            test_accuracy, test_loss = compute_accuracy_and_loss(model, val_ds, val_loader, device, loss_fn)

            if test_accuracy > max_test_acc[1]:
                max_test_acc[0] = epoch+1
                max_test_acc[1] = test_accuracy
            
            train_loss_lst.append(train_loss)
            train_acc_lst.append(train_accuracy)
            tset_loss_lst.append(test_loss)
            test_acc_lst.append(test_accuracy)
            
            print(f'Epoch {epoch+1}/{epochs}', end='')
            print(f'Train Acc: {train_accuracy:.5f}%' f' , Validation Acc: {test_accuracy:.5f}% |||| ', end='')
            print(f'Train Loss: {train_loss:.5f}' f' , Validation Loss: {test_loss:.5f}')

        tElapsed = datetime.now() - tStart
        print(f'(Elapsed time: {tElapsed})\n')

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            path = 'best.pt'
            torch.save(model, path)
            print('---------------------model save---------------------')

    print(best_accuracy)


if __name__ == '__main__':

    s = random.randint(0, 9999)
    random.seed(s)
    print(s)
    
    batch_size = 64
    dir_head = 'head_helmet_class/head'
    dir_helmet = 'head_helmet_class/helmet'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #model = ResNet152().to(device)
    model = ResNet34().to(device)
    epochs = 20
    transform = transforms.Compose(
    [transforms.Resize((224, 224)),
    #  transforms.Normalize((0.5077, 0.4813, 0.4312), (0.2000, 0.1986, 0.2034)),
     transforms.ToTensor()
     ])

    head_dataset = ImageClassificationDataset(dir_head, transform)
    helmet_dataset = ImageClassificationDataset(dir_helmet, transform)
    train_dataset, val_dataset, train_loader, val_loader = data_utils(head_dataset, helmet_dataset, batch_size)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.003)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
   
    train(device, model, epochs, train_dataset, train_loader, val_dataset, val_loader, loss_fn, optimizer, scheduler)
    print('Done!')

