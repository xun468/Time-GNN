# Time-GNN 
Pytorch implementation of TimeGNN: Temporal Dynamic Graph Learning for Time Series Forecasting. Full paper available <URL to arxiv>here 

<figure> 


## Requirements 
Libraries used for this project can be installed through ```pip install -r requirements.txt```

```
matplotlib==3.4.2
numpy==1.20.3
pandas==1.2.4
PyYAML==5.4.1
scikit_learn==0.24.2
torch==1.12.1
torch_geometric==2.1.0
torch_geometric_temporal==0.54.0
```

## Running
Overall experiment parameters are specified in Experiment_config.yaml. 
```
dataset: choose exchange, weather, solar, electricity, or traffic
features: whether to load as a single variate or multi variate time series; choose multi or single
seq_len: size of the window used for training
horizon: number of steps ahead to predict. Note: this project currently only supports seq2one forecasting
cut: percentage of dataset to use. Leave as null to use full dataset
runs: number of models to train 
n_epochs: number of epochs each model is trained
val_interval: number of epochs between validation runs 
output_dir: where the resulting models and evaluations are saved  
```

Models can be evaluated by simply running their respective training scripts (TimeGNN -> TimeGNN_train.py). Model hyperparameters are specified in these scripts using the args dictionary. The baseline training scripts used in the paper are in the baselines folder. Model architectures can be found in the models folder. 

### Adding New Datasets 
New datasets can be added by modifying utils/data_utils.py. Insert the file path and number of variables of the new dataset in dataset_loc and dataset_dims respectively. The read_data function assumes the dataset can be loaded into a dataframe where each column corresponds to a relevant variable and row to the timestep. The new dataset can then be chosen by modifying Experiment_config.yaml. 

## Citation 
<bibtex citation> 

