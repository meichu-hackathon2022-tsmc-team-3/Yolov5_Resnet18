import os
import sys
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import cv2
from PIL import Image
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from tqdm import tqdm

print(torch.__version__)


def yoloCutImage(model1, test_name_list, test_dir, crop_dir):

    print('Start testing...')

    transform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor()])

    model1.eval()
    with torch.no_grad():
        bar = tqdm(test_name_list)
        for _, name in enumerate(bar):
            
            # test image path
            image_path = os.path.join(test_dir, name)
            
            # 進行物件偵測
            results = model1(image_path)
            
            # return the predictions as a pandas dataframe
            bbox_df = results.pandas().xyxy[0]

            img = cv2.imread(image_path)
            show_img = cv2.imread(image_path)

            for bbox_number in range(len(bbox_df)):
            
                # 偵測到的bounding box
                xmin = int(bbox_df['xmin'][bbox_number])
                ymin = int(bbox_df['ymin'][bbox_number])
                xmax = int(bbox_df['xmax'][bbox_number])
                ymax = int(bbox_df['ymax'][bbox_number])
                confidence = str(round(bbox_df['confidence'][bbox_number],2))

                crop_img = []
                crop_img = img[ymin:ymax, xmin:xmax].copy()
                
                cv2.rectangle(show_img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.putText(show_img, confidence, (xmin-5, ymin-5), cv2.FONT_HERSHEY_DUPLEX,0.8, (0, 255, 0), 2, cv2.LINE_AA)
                
                # crop image path
                crop_name = name.split('.')[0] + '_' + str(bbox_number) + '.png'
                crop_image_path = os.path.join(crop_dir, crop_name)
                # 已經切好的暫存img
                cv2.imwrite(crop_image_path, crop_img)

                ################ 2nd stage #############################
                img = Image.open(crop_image_path).convert('RGB')
                image_tensor = transform(img)
                images = image_tensor.unsqueeze(0)
                images = images.to(device)
                outputs = model2(images)     
                _, predicted = torch.max(outputs.data, 1)
                pred = predicted.cpu().numpy()

            #y_true.extend(labels.cpu().numpy())
            outputs = model2(images) 
    
    print('The 1st stage finish !')


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
    print('The 2nd stage finish !')


if __name__ == '__main__':

    # test_dir = sys.argv[1]
    # crop_dir = sys.argv[2]
    test_dir = 'test/'
    crop_dir = 'crop/'

    # ############# 1st stage #################
    test_name_list = [s for s in os.listdir(test_dir)]

    model1 = torch.hub.load('yolov5.pt')

    yoloCutImage(model1, test_name_list, test_dir, crop_dir)
    
    ############# 2nd stage #################
    crop_name_list = [s for s in os.listdir(crop_dir)]
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    model2 = torch.load('Resnet18_98199.pt')
    
    netClassifier(device, model2, crop_name_list, crop_dir)

