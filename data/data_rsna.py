import torch
import pandas as pd
import numpy as np
import pydicom as com
from torch.utils.data import Dataset, dataloader

class rsna_dataset(Dataset):
    def __init__(self, data_df, transforms=None):
        super().__init__()
        df_z = data_df[0]
        df_o = data_df[1]

        npe = df_z.values()
        pe = df_o.values()
        self.input_paths = []
        self.labels = []
        for path in npe:
            self.input_paths.append(path)
            self.labels.append(0)
        
        for path in pe:
            self.input_paths.append(path)
            self.labels.append(1)
            
        self.input_paths = np.array(self.input_paths)
        self.labels = np.array(self.labels, dtype=np.float32)
    def __len__(self):
        return len(self.input_paths)
    
    def __getitem__(self, index):
        path = self.input_paths[index]
        label = self.labels[index]
        d_img = com.dcmread(path)
        arr_img = d_img.pixel_array
        arr_img = np.repeat(arr_img[np.newaxis, ...], 3, 0).astype(np.float32)
        arr_img /= arr_img.max()
        return torch.from_numpy(arr_img), torch.from_numpy(label)
        

def getLoaders(train, val, test, transforms=None):
    train_ds = rsna_dataset(train)
    val_ds = rsna_dataset(val)
    test_ds = rsna_dataset(test)

    train_loader = dataloader(train_ds, batch_size=32, shuffle=True)
    val_loader = dataloader(val_ds, batch_size=32, shuffle=True)
    test_loader = dataloader(test_ds, batch_size=32, shuffle=True)

    return train_loader, val_loader, test_loader

if __name__ == '__main__':
    train_csv = '' # get path
    df = pd.read_csv(train_csv)
    df_z = df.loc[df['pe_present_on_image'] == 0][:50000]
    df_o = df.loc[df['pe_present_on_image'] == 0][:50000]

    train_z = df_z.iloc[:40000]
    val_z = df_z.iloc[40000:45000]
    test_z = df_z.iloc[45000:]

    train_o = df_o.iloc[:40000]
    val_o = df_o.iloc[40000:45000]
    test_o = df_o.iloc[45000:]

    train_loader, val_loader, test_loader = getLoaders((train_z, train_o), (val_z, val_o), (test_z, test_o))
    batch_img, batch_label = next(iter(train_loader))

    fig = plt.figure(figsize=(4, 4))
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow(batch_img[i].permute(1, 2, 0))
        lbl = batch_label[i]
        title = 'PR present' if lbl else 'PE Absent'
        plt.title(title)
    plt.show()
