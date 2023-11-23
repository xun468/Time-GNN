import torch
import torch.optim as optim
import numpy as np
import os
import time
import gc
import yaml
import pickle
from contextlib import redirect_stdout
from utils.utils import *
from utils.data_utils import get_dataloaders, get_dataset_dims
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from models.TimeMTGNN import TimeMTGNN

#load experiment configs
with open('Experiment_config.yaml', 'r') as f:
    config = list(yaml.load_all(f))[0]
    
def train(model, model_type = ""):
    model.train()
    batch_losses = [] 

    for batch in train_loader:
        if config["features"] == "single":
            x_batch, y_batch = batch[0].float().to(device), batch[1].float().to(device)
            x_batch = x_batch.unsqueeze(1).unsqueeze(1)           
        else:
            x_batch, y_batch = batch[0].float().to(device), batch[1].float().to(device)
            x_batch = x_batch.permute(0,2,1)
            x_batch = x_batch.unsqueeze(1)
            y_batch = y_batch.unsqueeze(-1)
      
        # Make predictions
        y_hat = model(x_batch)     
        
        if config["features"] == "single":
            y_hat = y_hat.squeeze(1).squeeze(1)        
       
        # Computes loss
        loss = masked_mae(y_hat, y_batch)

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
        if config["features"] == "single":
            x_batch, y_batch = batch[0].float().to(device), batch[1].float().to(device)
            x_batch = x_batch.unsqueeze(1).unsqueeze(1)            
        else:
            x_batch, y_batch = batch[0].float().to(device), batch[1].float().to(device)
            x_batch = x_batch.permute(0,2,1)
            x_batch = x_batch.unsqueeze(1)
            y_batch = y_batch.unsqueeze(-1)            
      
        y_hat = model(x_batch)
        
        if config["features"] == "single":
            y_hat = y_hat.squeeze(1).squeeze(1)

        # Computes loss
        loss = masked_mae(y_hat, y_batch)     
        
        batch_val_losses.append(loss.item())
        
    return np.mean(batch_val_losses)

def test(test_loader, load_state = True, model_loc = ""):
    if load_state:
        model.load_state_dict(torch.load(model_loc))
      
    with torch.no_grad():
        model.eval()
        
        predictions = []
        values = []
        count = 0
        
        for batch in test_loader:
            if config["features"] == "single":
                x_batch, y_batch = batch[0].float().to(device), batch[1].float().to(device)
                x_batch = x_batch.unsqueeze(1).unsqueeze(1)
            else:
                x_batch, y_batch = batch[0].float().to(device), batch[1].float().to(device)
                x_batch = x_batch.permute(0,2,1)
                x_batch = x_batch.unsqueeze(1)
                y_batch = y_batch.unsqueeze(-1)
          
            y_hat = model(x_batch)

            if config["features"] == "single": 
                y_hat = y_hat.squeeze(1).squeeze(1)
            
            y_hat = y_hat.cpu().detach().numpy()
            predictions.append(y_hat)
            values.append(y_batch.cpu().detach().numpy())
            count += 1 
            if count == 50:
                break

    return predictions, values

#Recording args
model_type = "TimeMTGNN"
output_dir = config["output_dir"]
experiment_number = None
save_dir, model_dir = create_directories(model_type, output_dir, experiment_number)
print(save_dir)

#Dataset args    
input_dim, output_dim = get_dataset_dims(config["dataset"],config["features"])

#Load data
train_loader, val_loader, test_loader, test_loader_one, scaler = get_dataloaders(config["dataset"], seq_len = config["seq_len"], 
                                                                                 horizon = config["horizon"], features = config["features"], 
                                                                                 cut = config["cut"])
                                                                                 
#Model Args 
args = {
    "gcn_true" : True,
    "build_adj" : True,
    "gcn_depth" : 2,

    "kernel_set" : [2,3,6,7],
    "kernel_size" : 7,
    "dropout" : 0.3,
    "node_dim" : 40,
    "dilation_exponential" : 2,
    "conv_channels" : 8,
    "residual_channels" : 8,
    "skip_channels" : 16,
    "end_channels" : 32,

    "seq_length" : config["seq_len"],
    "in_dim" : 1,
    "out_dim" : 1,
    "num_nodes" : input_dim,
    "subgraph_size" : 20, #default 20

    "layers" : 5,
    "propalpha" : 0.05,
    "tanhalpha" : 3,
    "layer_norm_affline" : False
}
loss_fn = masked_mae
 
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
scaler = None
    
for i in range(0, config["runs"]):
    print("Run " + str(i))
    best_val = float('inf')
    
    model = None
    gc.collect()
    torch.cuda.empty_cache()

    model = TimeMTGNN(**args).to(device)
    optimizer = optim.Adam(model.parameters(), weight_decay = 1e-5)
    
    train_time = []    
    for epoch in range(config["n_epochs"]):
        t0 = time.time()
        batch_losses = train(model, loss_fn)
        t1 = time.time()
        
        train_losses.append(np.mean(batch_losses))
        train_time.append(t1-t0)

        if epoch % config["val_interval"] == 0:
            val_losses.append(val(model,loss_fn))
            val_epoch.append(epoch)
            print(
              f"[{epoch}/{n_epochs}] Training loss: {train_losses[-1]:.4f}\t Validation loss: {val_losses[-1]:.4f} \t Time: {t1-t0:.2f}"
          )

            if val_losses[-1] <= best_val and not np.isnan(val_losses[-1]):
                best_val = val_losses[-1]
                torch.save(model.state_dict(), model_dir + "best_run" + str(i) + ".pt")  
        else:
            print(f"[{epoch}/{n_epochs}] Training loss: {train_losses[-1]:.4f} \t Time: {t1-t0:.2f}")

    torch.save(model.state_dict(), model_dir + "last_run" + str(i) + ".pt")

    print("Last model this run: ")
    t0 = time.time()
    predictions, values = test(test_loader_one, load_state = False)
    t1 = time.time()
    inf_time = t1-t0
    metrics_last, df_results_last = metrics(predictions, values, metrics_best, scaler, test_loader_one.dataset.start, config["features"], train_time, inf_time)
    results_last.append(df_results_last)  

    print("")
    print("Best model this run: ")
    t0 = time.time()
    predictions, values = test(test_loader_one, load_state = True, model_loc = model_dir + "best_run" + str(i) + ".pt")
    t1 = time.time()    
    inf_time = t1-t0
    metrics_best, df_results_best = metrics(predictions, values, metrics_best, scaler, test_loader_one.dataset.start, config["features"], train_time, inf_time)

    results_best.append(df_results_best)
    print("")

print("DONE")
print(model_type)
print(config["dataset"])
print("horizon " + str(config["horizon"]))

print("\nBest Models Aggregate:")
print_metrics(metrics_best)
print("\nLast Models Aggregate:")
print_metrics(metrics_last)

with open(save_dir + 'out.txt', 'w') as f:
    with redirect_stdout(f):
        print(model_type)
        print(config["dataset"])
        print("horizon " + str(config["horizon"]))

        print("\nBest Models Aggregate:")
        print_metrics(metrics_best)
        print("\nLast Models Aggregate:")
        print_metrics(metrics_last)
        
with open(save_dir + model_type + "_best.pickle", 'wb') as handle:
    pickle.dump(results_best, handle)

with open(save_dir + model_type + "_last.pickle", 'wb') as handle:
    pickle.dump(results_last, handle)