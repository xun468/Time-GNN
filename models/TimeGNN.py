import numpy as np
import torch
import torch.nn as nn
from torch.nn import BatchNorm1d
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GatedGraphConv, GINConv, SAGEConv, GCN2Conv, SGConv, SAGEConv, SuperGATConv, GraphNorm, BatchNorm
from torch_geometric.utils import dense_to_sparse, dropout_adj
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                  enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                            dtype=np.int32)
    return labels_onehot
   
class TimeGNN(nn.Module):
    def __init__(self, loss, input_dim, hidden_dim, output_dim, seq_len, batch_size,
                 time_feature_type = "inception", edge_learning = "gumbel softmax", gtype = "sage", aggregate = "last",
                 keep_self_loops = False, enforce_consecutive = False, edge_dropout = False, 
                 block_size = 3, k = 5):        
        super(TimeGNN, self).__init__()
        print(gtype + str(block_size))
        
        self.loss = loss
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim        
        self.seq_len = seq_len 
        self.batch_size = batch_size 
        
        self.time_feature_type = time_feature_type
        self.edge_learning = edge_learning        
        self.gtype = gtype
        self.aggregate = aggregate 
        
        self.keep_self_loops = keep_self_loops
        self.enforce_consecutive = enforce_consecutive
        self.edge_dropout = edge_dropout 
        
        self.block_size = block_size        
        self.k = k            

        self.weighted = False
        if edge_learning in ["tanh", "gumbel weighted"]: 
            self.weighted = True

        #Feature Extraction
        #dialated inception
        if time_feature_type == "inception":        
            self.conv11 = nn.Conv1d(input_dim,hidden_dim,1, padding = "same")
            self.conv12 = nn.Conv1d(hidden_dim, hidden_dim,3, padding = "same", dilation=3)

            self.conv21 = nn.Conv1d(input_dim,hidden_dim,1, padding = "same")
            self.conv22 = nn.Conv1d(hidden_dim,hidden_dim,5, padding = "same", dilation=5)

            self.conv31 = nn.Conv1d(input_dim,hidden_dim,1, padding = "same")

            self.max_pool = nn.MaxPool1d(3, stride = 1)
            self.conv_final = nn.Conv1d(hidden_dim*3,hidden_dim, 5, padding = "same")


            self.fc_final = nn.Linear(hidden_dim*3, hidden_dim)

        elif time_feature_type == "wave":
            self.conv1 = nn.Conv1d(input_dim,hidden_dim,5, padding = "same")
            self.conv2 = nn.Conv1d(hidden_dim,hidden_dim,5, padding = "same", dilation=3)
            self.conv3 = nn.Conv1d(hidden_dim,hidden_dim,5, padding = "same", dilation=5)
            self.conv4 = nn.Conv1d(hidden_dim,hidden_dim,3, padding = "same", dilation = 7)
            self.bn1 = BatchNorm1d(hidden_dim)
            self.bn2 = BatchNorm1d(hidden_dim)
            self.bn3 = BatchNorm1d(hidden_dim)        
            self.bn4 = BatchNorm1d(hidden_dim)  

        else:
            assert False      
        
        #Edge Learning        
        ones = np.ones([seq_len, seq_len])
        self.rec_idx = torch.Tensor(np.array(encode_onehot(np.where(ones)[0]), dtype=np.float32)).to(device)
        self.send_idx = torch.Tensor(np.array(encode_onehot(np.where(ones)[1]), dtype=np.float32)).to(device)
        
        self.fc1 = torch.nn.Linear(hidden_dim*2, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 2)       
        
        #masks
        #triangle/time enforce mask
        self.tri_mask = torch.Tensor(np.tril(ones, k = -1)).bool().to(device)
        #diagonal/self loop mask 
        self.diagonal = torch.Tensor(np.diag(np.diag(ones))).bool().to(device)
        #force consecutive mask
        self.consecutive = torch.Tensor(np.eye(seq_len, seq_len, k = 1)).bool().to(device)
        #topk 
        self.k_mask = np.triu(ones, k = 1)
        for i in range(seq_len-k-1):
            self.k_mask[i] = 0
        self.k_mask = torch.Tensor(self.k_mask).bool().to(device)

        #Graph processing  
        self.gnns = nn.ModuleList()
        self.bns = nn.ModuleList()   
        for i in range(0, block_size):
            if gtype == "gcnconv":
                self.gnns.append(GCNConv(hidden_dim, hidden_dim).to(device))
            elif gtype == "sage":
                self.gnns.append(SAGEConv(in_channels = hidden_dim, out_channels = hidden_dim, normalize = False).to(device))
            elif gtype == "GAT":
                self.gnns.append(SuperGATConv(in_channels = -1, out_channels = hidden_dim, heads = 1))
            elif gtype == "gated":
                self.gnns.append(GatedGraphConv(out_channels = hidden_dim, num_layers = 5).to(device))
            elif gtype == "gcn2":
                self.gnns.append(GCN2Conv(hidden_dim, alpha = 0.1).to(device))
            else:
                print(gtype + "not found")
                assert False
            self.bns.append(BatchNorm(hidden_dim))
                
        self.gnn_weights = nn.Linear(block_size, 1, bias = True)        

        self.fc_extra = nn.Linear(hidden_dim, int(hidden_dim/2))
        self.bn4 = BatchNorm1d(int(hidden_dim/2))
        self.output = nn.Linear(int(hidden_dim/2), output_dim)        
        
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, data, return_graphs = False):
        if len(data.shape) == 2:
            data = data.unsqueeze(1)

        data = data.permute(0,2,1)
        batch_size, _, seq_len = data.shape

        #feature extraction
        if self.time_feature_type == "inception":
            x1 = self.conv11(data)
            x1 = self.conv12(x1)

            x2 = self.conv21(data)
            x2 = self.conv22(x2)

            x3 = self.conv31(data)

            x = torch.cat([x1,x2,x3], dim = 1)
            x = torch.transpose(x, 1,2)
            x = self.fc_final(x)
            x = F.relu(x)
        elif self.time_feature_type == "wave":
            x = self.conv1(data)
            x = F.relu(x)
            x = F.dropout(x,0.1)
            x = self.bn1(x)

            x = self.conv2(x)
            x = F.relu(x)
            x = F.dropout(x,0.1)
            x = self.bn2(x)

            x = self.conv3(x)
            x = F.relu(x)
            x = F.dropout(x,0.1)
            x = self.bn3(x)

            x = torch.transpose(x, 1,2)
        
        receivers = torch.bmm(self.rec_idx.repeat(batch_size,1,1), x)
        senders = torch.matmul(self.send_idx, x)
        edges = torch.cat([senders, receivers], dim=2)

        #edge learning
        edges = self.fc1(edges)
        edges = F.relu(edges)
        edges = self.fc2(edges)

        #gumbel learning
        if self.edge_learning == "gumbel softmax":
            adj = F.gumbel_softmax(edges, tau = 0.5, hard = True)
            adjs = []

            for i in range(batch_size):
                a = adj[i][:, 0].clone().reshape(self.seq_len, -1) #reshape into (seq len, seq len)

                #apply masks
                a = a.masked_fill_(self.tri_mask, 0)

                if self.enforce_consecutive == True: 
                    a = a.masked_fill_(self.consecutive, 1)
                if self.keep_self_loops == False: 
                    a = a.masked_fill_(self.diagonal, 0)

                adjs.append(a) 

        elif self.edge_learning == "gumbel topk":
            #gumbel topk learning 
            adj = F.gumbel_softmax(edges, tau = 0.5, hard = False)
            adjs = []

            for i in range(batch_size):
                a = adj[i][:, 0].clone().reshape(self.seq_len, -1) #reshape into (seq len, seq len)

                #apply masks
                a = a.masked_fill_(self.tri_mask, 0)
                if self.args.keep_self_loops == False: 
                    a = a.masked_fill_(self.diagonal, 0)

                s,t=a.topk(self.k,1)
                a = a.fill_(0)
                a = a.scatter_(1, t, s.fill_(1))

                a = a.masked_fill_(self.tri_mask, 0)
                a = a.masked_fill_(self.k_mask, 1)

                if self.args.enforce_consecutive == True: 
                    a = a.masked_fill_(self.consecutive, 1)

                adjs.append(a) 
        elif self.edge_learning == "gumbel weighted":
            adj = F.gumbel_softmax(edges,tau = 0.5, hard = False)
            adj = F.relu(adj)
            adjs = []

            for i in range(batch_size):
                a = adj[i][:, 0].clone().reshape(self.seq_len, -1) #reshape into (seq len, seq len)
                a[a < 0.5] = 0

                #apply masks
                a = a.masked_fill_(self.tri_mask, 0)

                if self.args.enforce_consecutive == True: 
                    a = a.masked_fill_(self.consecutive, 1)
                if self.args.keep_self_loops == False: 
                    a = a.masked_fill_(self.diagonal, 0)

                adjs.append(a) 

        elif self.edge_learning == "tanh": 
            adj = torch.tanh(edges)
            adj = F.relu(adj) 
            adjs = []

            for i in range(batch_size):
                a = adj[i][:, 0].clone().reshape(self.seq_len, -1) #reshape into (seq len, seq len)
                a[a < 0.5] = 0

                #apply masks
                a = a.masked_fill_(self.tri_mask, 0)

                if self.args.enforce_consecutive == True: 
                    a = a.masked_fill_(self.consecutive, 1)
                if self.args.keep_self_loops == False: 
                    a = a.masked_fill_(self.diagonal, 0)

                adjs.append(a)       

        adj = torch.stack(adjs, dim = 0)
      
        #GNNs
        edge_list, edge_weights = dense_to_sparse(adj)        
        x = x.reshape(-1, self.hidden_dim) #reshape into (all nodes, hidden dim) for pytorch geometric
            
        if self.edge_dropout:
                e,w = dropout_adj(edge_list, p = 0.1)
        else:
                e = edge_list
                w = edge_weights
                
        x_stack = [x]      
        for i in range(len(self.gnns)):
            gnn = self.gnns[i]
            bn = self.bns[i]            
                
            if self.weighted: 
                if self.gtype == "gcn2":
                    x = gnn(x = x, x_0 = x_stack[-1], edge_index = e, edge_weight = w)
                else:
                    x = gnn(x = x, edge_index = e, edge_weight = w)
            else:
                x = gnn(x, e)
            x = bn(x)
            x_stack.append(x)
      
        x = torch.stack(x_stack[1:], dim=-1)      
        x = self.gnn_weights(x).squeeze(-1)    
        x = x.reshape(batch_size, -1, self.hidden_dim) #reshape back into (batch size, seq len, hidden dim)
        x = torch.relu(x)

        if self.aggregate == "mean":
            x = torch.mean(x, dim=1)
        elif self.aggregate == "last":
            x = x[:,-1]

        x = self.fc_extra(x)
        x = F.relu(x)
        output = self.output(x)

        if return_graphs:
            return output, adj
        
        if self.output_dim > 1:
            return output.unsqueeze(1)
        
        return output
   
