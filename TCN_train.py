import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import os
import time
import yaml
import pickle
from utils import *
from data_utils import get_dataloaders, get_dataset_dims
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from models.TCN import TCN

#load experiment configs
with open('Experiment_config.yaml', 'r') as f:
    config = list(yaml.load_all(f))[0]
    
def train(model, model_type = ""):
    model.train()
    batch_losses = [] 

    for batch in train_loader:
        x_batch, y_batch = batch[0].float().to(device), batch[1].float().to(device)
        if config["features"] == "single":
            x_batch = x_batch.unsqueeze(1)
        else:
            x_batch = x_batch.permute(0,2,1)
            y_batch = y_batch.squeeze(1)
      
        # Make predictions
        y_hat = model(x_batch)        
       
        # Computes loss
        loss = model.loss(y_hat, y_batch)
        
        # Computes gradients
        loss.backward()

        # Updates parameters and zeroes gradients
        optimizer.step()
        optimizer.zero_grad()

        batch_losses.append(loss.item())
    
    return batch_losses 

def val(model, model_type = ""):
    model.eval()
    batch_val_losses = []    

    for batch in val_loader:
        x_batch, y_batch = batch[0].float().to(device), batch[1].float().to(device)
        if config["features"] == "single":
            x_batch = x_batch.unsqueeze(1)
        else:
            x_batch = x_batch.permute(0,2,1) 
            y_batch = y_batch.squeeze(1)
      
        y_hat = model(x_batch)

        # Computes loss
        loss = model.loss(y_hat, y_batch)
        
        batch_val_losses.append(loss.item())
        
    return np.mean(batch_val_losses)

def test(test_loader, load_state = True, model_loc = "", return_graphs = False):
    if load_state:
        model.load_state_dict(torch.load(model_loc))
    
    with torch.no_grad():
        model.eval()
        
        predictions = []
        values = []
        
        for batch in test_loader:
            x_batch, y_batch = batch[0].float().to(device), batch[1].float().to(device)
            if config["features"] == "single":
                x_batch = x_batch.unsqueeze(1)
            else:
                x_batch = x_batch.permute(0,2,1)
                y_batch = y_batch.squeeze(1)            
         
            y_hat = model(x_batch)
            
            y_hat = y_hat.cpu().detach().numpy()
            predictions.append(y_hat)
            values.append(y_batch.cpu().detach().numpy())
            
    print(predictions[0].shape)
    print(values[0].shape)

    return predictions, values

#Recording args
model_type = "TCN"
experiment_number = get_experiment_number()
save_dir = "OutputDump/experiment_" + str(experiment_number) + "/"
model_dir = save_dir + model_type + "_models/"
print(save_dir)

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)    
    
#Dataset args    
input_dim, output_dim = get_dataset_dims(config["dataset"],config["features"])

#Load data
train_loader, val_loader, test_loader, test_loader_one, scaler = get_dataloaders(config["dataset"], seq_len = config["seq_len"], 
                                                                                 horizon = config["horizon"], features = config["features"], 
                                                                                 cut = config["cut"])

#Model Args 
args = {
    "input_size" : input_dim, 
    "output_size": output_dim, 
    "num_channels": [128,128,128],
}

loss = nn.MSELoss()
 
#Training args
train_losses = [] 
val_losses = []
val_epoch = [] 
metrics_last = {}
metrics_best = {}
results_last = []
results_best = [] 

#Save args
with open(save_dir + model_type + "_args.yaml", 'w') as f:
    yaml.dump(args, f, sort_keys=False, default_flow_style=False)
    
with open(save_dir + "Experiment_config.yaml", "w") as f:
    yaml.dump(config, f, sort_keys=False, default_flow_style=False)
    
#Training
did_nan = False 
n_epochs = config["n_epochs"]
for i in range(0, config["runs"]):
    print("Run " + str(i))
    best_val = float('inf')

    model = TCN(loss = loss, **args).to(device)
    optimizer = optim.Adam(model.parameters())
    
    train_time = []    
    for epoch in range(n_epochs):
        t0 = time.time()
        batch_losses = train(model)
        t1 = time.time()
        
        train_losses.append(np.mean(batch_losses))
        train_time.append(t1-t0)

        if epoch % config["val_interval"] == 0:
            val_losses.append(val(model))
            val_epoch.append(epoch)
            print(
              f"[{epoch}/{n_epochs}] Training loss: {train_losses[-1]:.4f}\t Validation loss: {val_losses[-1]:.4f} \t Time: {t1-t0:.2f}"
          )

            if val_losses[-1] <= best_val and not np.isnan(val_losses[-1]):
                best_val = val_losses[-1]
                torch.save(model.state_dict(), model_dir + "best_run" + str(i) + ".pt")    

    torch.save(model.state_dict(), model_dir + "last_run" + str(i) + ".pt")

    print("Last")
    t0 = time.time()
    predictions, values = test(test_loader_one, load_state = False)
    t1 = time.time()
    inf_time = t1-t0
    metrics_last, df_results_last = metrics(predictions, values, metrics_best, scaler, test_loader_one.dataset.start, config["features"], train_time, inf_time)
    results_last.append(df_results_last)  

    print("")
    print("Best")
    t0 = time.time()
    predictions, values = test(test_loader_one, load_state = True, model_loc = model_dir + "best_run" + str(i) + ".pt")
    t1 = time.time()
    
    inf_time = t1-t0
    metrics_best, df_results_best = metrics(predictions, values, metrics_best, scaler, test_loader_one.dataset.start, config["features"], train_time, inf_time)

    results_best.append(df_results_best)
    print("")

print("DONE")
print(model_type)
print("horizon " + str(config["horizon"]))

print("Best:")
print_metrics(metrics_best)
print("Last:")
print_metrics(metrics_last)

if config["features"] == "single":
    print_graphs(results_best, title = model_type)    
    
with open(save_dir + model_type + "_best.pickle", 'wb') as handle:
    pickle.dump(results_best, handle)

with open(save_dir + model_type + "_last.pickle", 'wb') as handle:
    pickle.dump(results_last, handle)