import numpy as np
import os
import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import PIL 
import matplotlib.pyplot as plt

class rsna_data(Dataset):
    def __init__(self,data_file, transform=None):
        self.df_zero = data_file[0]
        self.df_ones = data_file[1]
        self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def mod_label(self, label): # This function adds a 15th item for 'No Findings' for all labels
        mod_lbl = []
        flag = 0
        for ele in label:
            if ele:
                flag = 1
                break
        if flag:
            mod_lbl = np.append(label, [0])
        else:
            mod_lbl = np.append(label, [1])
        return mod_lbl
    
    def __getitem__(self, index):
        image_path = os.path.join(self.root_dir, self.data[index][0])
        #print(self.data[index][0])
        image = PIL.Image.open(image_path).convert('RGB')
        label = self.data[index][1].astype('float32')
        #print(label)
        label = self.mod_label(label)
        #print(label)
        target = torch.zeros(len(label))
        target[label == 1] = 1  
        
        
        if self.transform is not None:
            image = self.transform(image)
        
        return image, target
    
def textfile(data_file):
    label=[]
    # read the text file and split the image filename and its corresponding labels
    with open(data_file, 'r') as f:
        lines = f.readlines()
    for i in range(len(lines)):
        dummy = []
        image_label_pair = lines[i].strip().split(' ')
        image_filename = image_label_pair[0]
        image_label = np.array(image_label_pair[1:], dtype=np.int32)
        dummy = [image_filename,image_label]
        label.append(dummy)
    
    return label


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])



def getdataset(base_path,tr_label,val_label,ts_label):
    train_dataset = ChestXrayDataset(root_dir=base_path, data_file=tr_label, transform=transform)
    test_dataset =ChestXrayDataset(root_dir=base_path, data_file=ts_label, transform=transform)
    val_dataset = ChestXrayDataset(root_dir=base_path, data_file=val_label, transform=transform)
    return  train_dataset,test_dataset,val_dataset

def getdataloader(train_dataset,test_dataset,val_dataset,batch_size,num_gpus):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=num_gpus)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,num_workers=num_gpus)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True,num_workers=num_gpus)
    return train_loader,test_loader,val_loader

def getClasses():
    classes = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration' , 'Mass', 'Nodule', 'Pneumonia', 
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia', 'No_Findings']
    return classes


# Load the train and test data files
# train_data_file = '../swin_transformer/Dataset/Xray14_train_official.txt'
# test_data_file = '../swin_transformer/Dataset/Xray14_test_official.txt'

# # Load the train and test datasets
# train_dataset = ChestXrayDataset(root_dir='../swin_transformer/Dataset/images/', data_file=train_data_file, transform=transform)
# test_dataset = ChestXrayDataset(root_dir='../swin_transformer/Dataset/images/', data_file=test_data_file, transform=transform)

# # Create DataLoaders for the train and test datasets
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
if __name__ == '__main__':
    base_path = r'../Images/images'
    tr_label = r'../swin_transformer/Practice/Dataset/Xray14_train_official.txt'
    val_label = r'../swin_transformer/Practice/Dataset/Xray14_val_official.txt'
    ts_label = r'../swin_transformer/Practice/Dataset/Xray14_test_official.txt'
    classes = getClasses()
    print(classes)
    print(len(classes))
    train_dataset,val_dataset,test_dataset = getdataset(base_path,tr_label,val_label,ts_label)
    print(len(train_dataset),len(val_dataset),len(test_dataset))
    train_loader,test_loader,val_loader = getdataloader(train_dataset,val_dataset,test_dataset,32)
    
    batch_img, batch_label = next(iter(train_loader))
    print(f'Label dimensions: {batch_label.shape}')
    fig = plt.figure(figsize=(4, 4))
    for i in range(0,4):
        indices = torch.nonzero(batch_label[i][0:15] == 1).squeeze(0)
        #print(indices)
        class_labels = [classes[idx] for idx in indices if indices.dim() > 0]  
        title = ', '.join(class_labels)
        #plt.subplot(2, 2, i + 1)
        plt.imshow(batch_img[i].permute(1,2,0),cmap='gray')
        plt.title(f"{batch_label[i]}\n {title}")
        plt.show()