import torch
import torch.optim as optim
import numpy as np
import os
import time
import yaml
import pickle
from utils import *
from data_utils import get_dataloaders, get_dataset_dims
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from models.GTS.model import GTSModel
from models.GTS.loss import masked_mae_loss, masked_mape_loss, masked_rmse_loss, masked_mse_loss

#load experiment configs
with open('Experiment_config.yaml', 'r') as f:
    config = list(yaml.load_all(f))[0]
    
def train(model, batches_seen):    
    label = 'with_regularization'
    batch_losses = [] 
    
    
    for batch in train_loader:
        optimizer.zero_grad()
        x_batch = batch[0].unsqueeze(-1)
        y_batch = batch[1].unsqueeze(-1)
        x_batch = x_batch.permute(1,0,2,3)
        y_batch = y_batch.permute(1,0,2,3)
        batch_size = x_batch.shape[1]
        x_batch = x_batch.view(config["seq_len"], batch_size, args["num_nodes"]*args["input_dim"]).to(device)
        y_batch = y_batch[..., :args["output_dim"]].view(args["horizon"], batch_size, 
                                                                args["num_nodes"]*args["output_dim"]).to(device)
        x_batch = x_batch.float()
        y_batch = y_batch.float()
#         print(x_batch.type())
#         print(y_batch.type())
        
        y_hat, mid_output = model(label, x_batch, train_feas, args['temp'], args['gumbel_soft'], y_batch, batches_seen)
        
        loss_1 = gts_loss(y_batch, y_hat)
        pred = mid_output.view(mid_output.shape[0] * mid_output.shape[1])
        true_label = adj_mx.view(mid_output.shape[0] * mid_output.shape[1]).to(device)
        loss2 = torch.nn.BCELoss()
        loss_g = loss2(pred, true_label)
        loss = loss_1 + loss_g
        batch_losses.append(loss.item())
        
        loss.backward()  
        optimizer.step()
        # gradient clipping - this does it in place
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        batches_seen+=1
        
    return batch_losses, batches_seen

def val(model):
    label = 'with_regularization'
    model.eval()
    batch_val_losses = []   
    
    for batch in val_loader:
        x_batch = batch[0].unsqueeze(-1)
        y_batch = batch[1].unsqueeze(-1)
        x_batch = x_batch.permute(1,0,2,3)
        y_batch = y_batch.permute(1,0,2,3)
        batch_size = x_batch.shape[1]
        x_batch = x_batch.view(args["seq_len"], batch_size, args["num_nodes"]*args["input_dim"]).to(device)
        y_batch = y_batch[..., :args["output_dim"]].view(args["horizon"], batch_size, 
                                                                args["num_nodes"]*args["output_dim"]).to(device)
        x_batch = x_batch.float()
        y_batch = y_batch.float()
        
        y_hat, mid_output = model(label, x_batch, train_feas, args['temp'], args['gumbel_soft'])
        
        loss_1 = gts_loss(y_batch, y_hat)
        pred = torch.sigmoid(mid_output.view(mid_output.shape[0] * mid_output.shape[1]))
        true_label = adj_mx.view(mid_output.shape[0] * mid_output.shape[1]).to(device)
        loss2 = torch.nn.BCELoss()
        loss_g = loss2(pred, true_label)
        loss = loss_1 + loss_g
        batch_val_losses.append(loss.item())
        
    return np.mean(batch_val_losses)

def test(model, test_loader, model_loc = "", load_state = True):
    label = 'with_regularization'
    if load_state:
        model.load_state_dict(torch.load(model_loc))
    
    with torch.no_grad():
        model.eval()        
        predictions = []
        values = []
        
        for batch in test_loader:
            x_batch = batch[0].unsqueeze(-1)
            y_batch = batch[1].unsqueeze(-1)
            x_batch = x_batch.permute(1,0,2,3)
            y_batch = y_batch.permute(1,0,2,3)
            batch_size = x_batch.shape[1]
            x_batch = x_batch.view(args["seq_len"], batch_size, args["num_nodes"]*args["input_dim"]).to(device)
            y_batch = y_batch[..., :args["output_dim"]].view(args["horizon"], batch_size, 
                                                                    args["num_nodes"]*args["output_dim"]).to(device)
            x_batch = x_batch.float()
            y_batch = y_batch.float()

            y_hat, mid_output = model(label, x_batch, train_feas, args['temp'], args['gumbel_soft'])
            predictions.append(y_hat.cpu().detach().numpy())
            values.append(y_batch.cpu().detach().numpy())
            
    print(predictions[0].shape)
    print(values[0].shape)

    return predictions, values

#Recording args
model_type = "GTS"
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
#Model args
args = {
  "cl_decay_steps": 2000,
  "filter_type": "dual_random_walk",
  "horizon": 1,
  "input_dim": 1,
  "l1_decay": 0,
  "max_diffusion_step": 2,
  "num_nodes": input_dim,
  "num_rnn_layers": 1,
  "output_dim": 1,
  "rnn_units": 128,
  "seq_len": config["seq_len"],
  "use_curriculum_learning": True,
  "dim_fc": 84672,
  "temp" : 0.5,
  "gumbel_soft" : True,
  "knn_k": 4
}

def gts_loss(y_true, y_predicted):
        return masked_mae_loss(y_predicted, y_true)

loss = gts_loss  

#Train args 
runs = 2
n_epochs = 5
val_interval = 1 

train_losses = [] 
val_losses = []
val_epoch = [] 

metrics_last = {}
metrics_best = {}
results_last = []
results_best = [] 

from sklearn.neighbors import kneighbors_graph
train_feas = train_loader.dataset.data
g = kneighbors_graph(train_feas.T, args.get('knn_k'), metric='cosine')

g = np.array(g.todense(), dtype=np.float32)

train_feas = torch.Tensor(train_feas).to(device)
adj_mx = torch.Tensor(g).to(device)

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
    batches_seen = 0

    model = GTSModel(0.5, **args).to(device)
    optimizer = optim.Adam(model.parameters(), weight_decay = 1e-5)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 30, 40], gamma=float(0.1))
    
    train_time = []    
    for epoch in range(config["n_epochs"]):
        t0 = time.time()
        batch_losses, batches_seen = train(model, batches_seen)
        t1 = time.time()
        lr_scheduler.step()

        train_losses.append(np.mean(batch_losses))
        train_time.append(t1-t0)
        train_losses.append(np.mean(batch_losses))

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
    predictions, values = test(model, test_loader_one, load_state = False)
    t1 = time.time()
    inf_time = t1-t0
    metrics_last, df_results_last = metrics(predictions, values, metrics_best, scaler, test_loader_one.dataset.start, config["features"], train_time, inf_time)
    results_last.append(df_results_last)  

    print("")
    print("Best")
    t0 = time.time()
    predictions, values = test(model, test_loader_one, load_state = True, model_loc = model_dir + "best_run" + str(i) + ".pt")
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