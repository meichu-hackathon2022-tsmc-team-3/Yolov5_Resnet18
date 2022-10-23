import os
import sys
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import cv2
from PIL import Image
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm


print(torch.__version__)


def featureMap(device, model, crop_name_list, crop_dir):
    

    # we will save the conv layer weights in this list
    model_weights =[]
    #we will save the 49 conv layers in this list
    conv_layers = []
    # get all the model children as list
    model_children = list(model.resnet152.children())
    print(model_children)
    #counter to keep count of the conv layers
    counter = 0
    #append all the conv layers and their respective wights to the list
    for i in range(len(model_children)):
        if type(model_children[i]) == nn.Conv2d:
            counter+=1
            model_weights.append(model_children[i].weight)
            conv_layers.append(model_children[i])
        elif type(model_children[i]) == nn.Sequential:
            for j in range(len(model_children[i])):
                for child in model_children[i][j].children():
                    if type(child) == nn.Conv2d:
                        counter+=1
                        model_weights.append(child.weight)
                        conv_layers.append(child)
    model = model.to(device)


    transform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor()])


    with torch.no_grad():
        bar = tqdm(crop_name_list)
        for _, name in enumerate(bar):

            image_path = os.path.join(crop_dir, name)
            img = Image.open(image_path).convert('RGB')
            image_tensor = transform(img)
            images = image_tensor.unsqueeze(0)
            image = images.to(device)

            outputs = []
            names = []
            print(conv_layers)
            for layer in conv_layers[0:]:
                image = layer(image)
                print(image)
                outputs.append(image)
                names.append(str(layer))

            processed = []
            for feature_map in outputs:
                feature_map = feature_map.squeeze(0)
                gray_scale = torch.sum(feature_map,0)
                gray_scale = gray_scale / feature_map.shape[0]
                processed.append(gray_scale.data.cpu().numpy())

            print(processed)

            fig = plt.figure(figsize=(30, 50))
            for i in range(len(processed)):
                a = fig.add_subplot(5, 4, i+1)
                imgplot = plt.imshow(processed[i])
                a.axis("off")
                a.set_title(names[i].split('(')[0], fontsize=30)
            plt.savefig(str('feature_maps.jpg'), bbox_inches='tight')

        

    print('Done !')



def netClassifier(device, model2, crop_name_list, crop_dir):
    
    if not os.path.exists(crop_dir):
        os.makedirs(crop_dir)

    print('Starting the 2nd stage...')

    transform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor()])

    correct = 0
    y_true = []
    y_pred = []

    model2.eval()

    with torch.no_grad():
        bar = tqdm(crop_name_list)
        for _, name in enumerate(bar):

            image_path = os.path.join(crop_dir, name)
            img = Image.open(image_path).convert('RGB')
            image_tensor = transform(img)
            images = image_tensor.unsqueeze(0)
            images = images.to(device)

            #y_true.extend(labels.cpu().numpy())
            outputs = model2(images)     
            _, predicted = torch.max(outputs.data, 1)
            y_pred.extend(predicted.cpu().numpy())
            #correct += (predicted == labels).sum()

    print(crop_name_list)
    print('predict: ', y_pred)
    # print(f"Accuracy Score: {accuracy_score(y_true, y_pred)*100:.3f}% ||| ", end='')
    # print(f"Precision Score: {precision_score(y_true, y_pred)*100:.3f}% ||| ", end='')
    # print(f"Recall Score: {recall_score(y_true, y_pred)*100:.3f}% ||| ", end='')
    # print(f"F1 Score: {f1_score(y_true, y_pred)*100:.3f}%")
    # print(f"Confusion Matrix:\n {confusion_matrix(y_true, y_pred)}")
    print('Done !')


if __name__ == '__main__':

    # test_dir = sys.argv[1]
    # crop_dir = sys.argv[2]
    test_dir = 'test/'
    crop_dir = 'crop/'

    crop_name_list = [s for s in os.listdir(crop_dir)]
    print(crop_name_list)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    model2 = torch.load('Resnet18_98199.pt')
    
    featureMap(device, model2, crop_name_list, crop_dir)

