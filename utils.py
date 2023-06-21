import matplotlib.pyplot as plt
from statistics import mean, stdev
import numpy as np
import pandas as pd
import torch
import os

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def shift_loss(outputs, targets, history, device, a1=0.5, penalize_all_steps=False):
    targets = targets.to(device)
    mse_loss = (outputs-targets.squeeze(-1))**2

    # keep the feature to be predicted in case of multivariate time series
    if history.shape[-1]>1:
        history = history[:,:,-1].unsqueeze(-1)

    # Penalty loss for 1-step ahead forecasting
    # targets of shape [batch_size,1], outputs [batch_size,1], history [batch_size,k-window,1] (select t-1 so shape becomes [batch_size,1]) - loss of size [batch_size,1]
    if not penalize_all_steps and outputs.shape[-1]==1:  
        # penalize t and t-1      
        a0=1
        penalty_term = torch.mul((targets.squeeze(-1) - history[:,-1,:]), (targets.squeeze(-1) - outputs))**2
        loss = a0*mse_loss + a1*penalty_term
        #print('Penalty', penalty_term.mean())
        #print('mse', (torch.abs((outputs-targets.squeeze(-1)))**2).mean())
        a_term = a1
    elif penalize_all_steps and outputs.shape[-1]==1:
        # penalize t and t-1, t-2, ..., (length of history window)/4 
        a0=1
        loss = a0*mse_loss
        penalty_term_all = 0
        a_term = []
                
        for i in range(int(history.size()[1]/4), history.size()[1]):
            penalty_term = torch.mul((targets.squeeze(-1) - history[:,-1,:]), (targets.squeeze(-1) - outputs))**2
            loss = loss + a1*penalty_term
            penalty_term_all = penalty_term_all + penalty_term
            a_term.append(a1)

    return loss.mean()
    
def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)
    
def inverse_transform(scaler, df, columns):
    if scaler is not None: 
        for col in columns:
            # df[col] = scaler.inverse_transform(df[col])
            df[col] = df[col].map(scaler.inverse_transform)
    return df

def format_predictions(predictions, values, index_start, features):
    if features == "multi":
        preds, vals = [], []
        for i in range(len(predictions)):
            preds.append(predictions[i].flatten())
            vals.append(values[i].flatten())
    else:
        vals = np.concatenate(values, axis=0).ravel()
        preds = np.concatenate(predictions, axis=0).ravel()

    df_result = pd.DataFrame(data={"value": vals, "prediction": preds})
    df_result.index = df_result.index + index_start
    
    return df_result

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, f1_score

def calculate_metrics(df):
     return {'mae' : mean_absolute_error(df.value.tolist(), df.prediction.tolist()),
             'mse' : mean_squared_error(df.value.tolist(), df.prediction.tolist()),
             'r2' : r2_score(df.value.tolist(), df.prediction.tolist(), multioutput = "variance_weighted"),}

def metrics(predictions, values, total_metrics, scaler, index_start, features, train_time, inf_time):  
    df_result = format_predictions(predictions, values, index_start, features)

    if df_result['prediction'].isnull().values.any():
        print("NaNs in predictions")
    print(df_result.head(1))

    result_metrics = calculate_metrics(df_result)    

    result_metrics['train time (s)'] = np.mean(train_time)
    result_metrics['inference time (ms)'] = (inf_time/len(predictions)) * 1000
    
    print(result_metrics)
    for metric in result_metrics.keys():
        if metric in total_metrics.keys():
            total_metrics[metric].append(result_metrics[metric])
        else: 
            total_metrics[metric] = [result_metrics[metric]]

    return total_metrics, df_result

def print_metrics(total_metrics):
    mprint = []
    for metric in total_metrics.keys(): 
        if isinstance(total_metrics[metric], np.float64):
            mprint.append(f'{total_metrics[metric]:.3f}')
        else:
            print(metric + " " + f'{mean(total_metrics[metric]):.3f} ± {np.std(total_metrics[metric], ddof=1):.3f}')
            mprint.append(f'{mean(total_metrics[metric]):.3f} ± {np.std(total_metrics[metric], ddof=1):.3f}')      
    print(','.join(map(str,mprint)))

def print_graphs(df_results, savedir, title = ""):  
    for n in range(0,len(df_results)):
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(13.2,4), gridspec_kw={'height_ratios': [3, 1]})
        df_results[n]['error'] = df_results[n]['value'] - df_results[n]['prediction']
        df_result_small = df_results[n][0:100]


        axes[0][0].set_title(title)
        axes[0][0].plot(df_result_small['value'], color = '#1f77b4', alpha = 0.8, label = "value")
        axes[0][0].plot(df_result_small['prediction'], color = '#ff7f0e', alpha = 0.8, label = "prediction")
        axes[0][0].legend()
        axes[1][0].plot(df_result_small['error'], color = '#d62728', alpha = 0.8)
        axes[1][0].get_xaxis().set_visible(False)
        axes[1][0].set_ylabel("Error")


        axes[0][1].set_title(title)
        axes[0][1].plot(df_results[n]['value'], color = '#1f77b4', alpha = 0.8, label = "value")
        axes[0][1].plot(df_results[n]['prediction'], color = '#ff7f0e', alpha = 0.8, label = "prediction")
        axes[0][1].legend()
        axes[1][1].plot(df_results[n]['error'], color = '#d62728', alpha = 0.8)
        axes[1][1].get_xaxis().set_visible(False)
        axes[1][1].set_ylabel("Error")        
                        
        plt.savefig(save_dir + "fig-" + str(n) + ".svg", bbox_inches='tight', format = "svg")
        plt.savefig(save_dir + "fig-" + str(n) + ".png", bbox_inches='tight', format = "png")
        plt.show()
    

def get_experiment_number():
    print(os.listdir("OutputDump/"))
    folders = [int(name.split("_")[1]) for name in os.listdir("OutputDump/") if os.path.isdir("OutputDump/" + name) and "experiment" in name]
    folders.sort()
    
    if len(folders) == 0:
        return 0
    return folders[-1] + 1
