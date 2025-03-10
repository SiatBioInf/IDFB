import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import torch.utils.data as Data

'''
    生成训练数据集
'''

def load_data(gpls=['GPL570', 'GPL20301', 'GPL24676']):
    dataset = [] # 数据集列表  
    for gpl in gpls:
        df = pd.read_csv(f'../Dataset/Pseudo data/{gpl}/mixed_samples.csv')
        try:
            df['GPL'] = gpls.index(gpl)
        except:
            df['GPL'] = len(gpls)
        dataset.append(df)
    return pd.concat(dataset, axis=0, ignore_index=True)


def gen_dataloader(batch_size):
    df = load_data()
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    # split the data into train and test
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y)

    # convert the data into tensor
    x_train = torch.tensor(x_train).float()
    x_test = torch.tensor(x_test).float()
    y_train = torch.tensor(y_train).long()
    y_test = torch.tensor(y_test).long()

    # create the dataset
    train_dataset = Data.TensorDataset(x_train, y_train)
    test_dataset = Data.TensorDataset(x_test, y_test)

    # create the dataloader
    train_dataloader = Data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = Data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader, test_dataloader

if __name__ == '__main__':
    train_dataloader, test_dataloader = gen_dataloader(64)
    for x, y in train_dataloader:
        print(x.shape)
        print(y.shape)
        break


