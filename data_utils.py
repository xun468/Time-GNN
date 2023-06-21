import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd 

dataset_loc = {
    "min-temps" : "datasets/daily-min-temperatures.csv",
    "electricity" : "datasets/ECL.csv",
    "solar" : "datasets/solar_AL.txt",
    "traffic" : "datasets/traffic.txt", 
    "exchange" : "datasets/exchange_rate.txt", 
    "weather" : "datasets/WTH.csv",
}

class StandardScaler():
    def __init__(self):
        self.mean = 0.
        self.std = 1.
    
    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)

    def transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / std

    def inverse_transform(self, data):      
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std

#         print(data.shape)
#         print(mean.shape)
#         print(std.shape)
        # if data.shape[-1] != mean.shape[-1]:
        #     mean = mean[-1:]
        #     std = std[-1:]
        return (data * std) + mean

def read_data(dataset, features, seq_len, target = "", scale = True, cut = None):
    df = pd.read_csv(dataset_loc[dataset])
    scaler = None
  
    if cut: 
        end = int(len(df) * cut) 
        df = df[:end]
  
    print(len(df))

    n_train = int(len(df) * 0.7)
    n_test = int(len(df) * 0.2)
    n_val = len(df) - (n_train + n_test)


    train_begin = 0 
    train_end = n_train

    test_begin = len(df) - n_test - seq_len
    test_end = len(df)

    val_begin = n_train - seq_len
    val_end = n_train + n_val

  
    if features == "single":
        if target: 
            df = df[[target]]
        else:
            df = df[df.columns[-1]]
    if features == "multi":
        if dataset in ['electricity', 'weather']:
            df = df[df.columns[1:]]
        else:
            df = df[df.columns[:]]
  
    if scale: 
        scaler = StandardScaler()
        train_data = df[0:n_train]
        scaler.fit(train_data.values)
        data = scaler.transform(df.values)
    else:
        data = df.values

    return data[train_begin:train_end], data[test_begin:test_end], data[val_begin:val_end], scaler, [train_begin, test_begin, val_begin]

class seq_data(Dataset):
    def __init__(self, data, start, seq_len = 20, horizon = 1, args = None):
        self.data = data
        self.seq_len = seq_len 
        self.horizon = horizon
        self.start = start   
        self.mode = "single-step"

    def __getitem__(self, index):
        seq_begin = index 
        seq_end = index + self.seq_len
        label_end = seq_end + self.horizon

        if self.mode == "single-step":
            label_begin = seq_end + self.horizon - 1

        else:
            label_begin = seq_end

        return self.data[seq_begin:seq_end], self.data[label_begin: label_end]

    def __len__(self):
        return len(self.data) - self.seq_len - self.horizon + 1

def get_dataloaders(dataset, batch_size = 16, seq_len = 20, horizon = 1, features = "single", 
                    target = "", scale = True, cut = None, args = None):
    assert dataset in dataset_loc.keys()
    assert features in ["single", "multi"]   
    print(dataset + " " + features)  
    
    train, test, val, scaler, starts = read_data(dataset, features, seq_len, target, cut = cut)

    train_data = seq_data(train, starts[0], seq_len, horizon, args)
    test_data = seq_data(test, starts[1], seq_len, horizon, args)
    val_data = seq_data(val, starts[2], seq_len, horizon, args)
    
    print(train_data.data[0].shape)

    train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True, drop_last = True)
    val_loader = DataLoader(val_data, batch_size = batch_size, shuffle = False, drop_last = True)
    test_loader = DataLoader(test_data, batch_size = batch_size, shuffle = False, drop_last = True)
    test_loader_one = DataLoader(test_data, batch_size = 1, shuffle = False, drop_last = False)

    return train_loader, val_loader, test_loader, test_loader_one, scaler

dataset_dims = {
    "electricity" : 321,
    "weather" : 12,
    "exchange" : 8
}

def get_dataset_dims(dataset, mode):
    if mode == "single":
        return 1, 1
    elif mode == "multi":
        return dataset_dims[dataset], dataset_dims[dataset]
    else: 
        print("Invalid feature mode " + mode)
        assert False