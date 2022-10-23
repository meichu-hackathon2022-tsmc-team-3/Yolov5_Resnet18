import os
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split, ConcatDataset
from torchvision import transforms
import torch


class ImageClassificationDataset(Dataset):

    def __init__(self, dir, transform=None, is_test=False):

        self.dir = dir
        self.img = [s for s in os.listdir(self.dir)]
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.img)

    def __getitem__(self, index):

        image_path = os.path.join(self.dir, self.img[index])
        image = Image.open(image_path).convert('RGB')
        
        if self.transform is not None:

            image = self.transform(image)
            
            if not self.is_test:
                if self.dir.split('/')[1] == 'head':
                    label = 0
                elif self.dir.split('/')[1] == 'helmet':
                    label = 1            

        return image, label


def data_utils(head_dataset, helmet_dataset, batch_size=32, ratio=0.9):

    head_train_size = int(len(head_dataset) * ratio)
    head_val_size = len(head_dataset) - head_train_size
    head_train_dataset, head_val_dataset = random_split(head_dataset, [head_train_size, head_val_size])
    # print(len(head_train_dataset))
    # print(len(head_val_dataset))

    helmet_train_size = int(len(helmet_dataset) * ratio)
    helmet_val_size = len(helmet_dataset) - helmet_train_size
    helmet_train_dataset, helmet_val_dataset = random_split(helmet_dataset, [helmet_train_size, helmet_val_size])
    # print(len(helmet_train_dataset))
    # print(len(helmet_val_dataset))

    train_dataset = ConcatDataset([head_train_dataset, helmet_train_dataset])
    val_dataset = ConcatDataset([head_val_dataset, helmet_val_dataset])

    train_loader = DataLoader(train_dataset, batch_size, num_workers=os.cpu_count(), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size, num_workers=os.cpu_count(), shuffle=True)

    return train_dataset, val_dataset, train_loader, val_loader


def get_stats(dataloader):
    tot_pixels = 0
    num_samples = 0
    mean = torch.empty(3)
    stdev = torch.empty(3)

    for data in dataloader:

        #data = data[0]
        b, c, h, w = data.shape
        num_pixels = b * h * w
        sum_ = torch.sum(data, dim=[0, 2, 3])
        sum_of_square = torch.sum(data ** 2, dim=[0, 2, 3])
        mean = (tot_pixels * mean + sum_) / (tot_pixels + num_pixels)
        stdev = (tot_pixels * stdev + sum_of_square) / (tot_pixels + num_pixels)
        num_samples += 1
        tot_pixels += num_pixels
        print('\r'+f'{(num_samples / len(dataloader)*100):.3f}% processed',end='')

    return mean, torch.sqrt(stdev - mean ** 2)


# if __name__ == '__main__':

#     dir_head = 'head_helmet_class/head'
#     dir_helmet = 'head_helmet_class/helmet'
    
#     transform = transforms.Compose(
#     [transforms.Resize((224, 224)),
#     #  transforms.Normalize((0.5077, 0.4813, 0.4312), (0.2000, 0.1986, 0.2034)),
#      transforms.ToTensor()
#      ])

#     # 128265
#     head_dataset = ImageClassificationDataset(dir_head, transform)
#     # 62279
#     helmet_dataset = ImageClassificationDataset(dir_helmet, transform)
#     # 171489(115438+56051), 19055(12827+6228)
#     train_dataset, val_dataset, train_loader, val_loader = data_utils(head_dataset, helmet_dataset)
    
#     # print(len(train_dataset))
#     # print(len(val_dataset))
#     # print(train_dataset[-1])