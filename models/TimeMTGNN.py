from __future__ import division

import numbers
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GatedGraphConv, GINConv, SAGEConv, GCN2Conv, SGConv, SAGEConv, SuperGATConv, GraphNorm, BatchNorm
from torch_geometric.utils import dense_to_sparse, dropout_adj
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                  enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                            dtype=np.int32)
    return labels_onehot

class TimeGraphConstructorMTGNN(nn.Module):
  def __init__(self, input_dim, hidden_dim, seq_len):
    super(TimeGraphConstructorMTGNN, self).__init__()
    self.conv11 = nn.Conv1d(input_dim,hidden_dim,1, padding = "same")
    self.conv12 = nn.Conv1d(hidden_dim, hidden_dim,3, padding = "same", dilation=3)

    self.conv21 = nn.Conv1d(input_dim,hidden_dim,1, padding = "same")
    self.conv22 = nn.Conv1d(hidden_dim,hidden_dim,5, padding = "same", dilation=5)

    self.conv31 = nn.Conv1d(input_dim,hidden_dim,5, padding = "same")

    self.fc_final = nn.Linear(hidden_dim*3, hidden_dim)

    self.seq_len = seq_len

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

  def forward(self, data: torch.FloatTensor)  -> torch.FloatTensor:
    data = data.squeeze(1)
    data = data.float()

    batch_size, _, seq_len = data.shape
    # print(data.shape)
    # print(data.dtype)

    x1 = self.conv11(data)
    x1 = self.conv12(x1)

    x2 = self.conv21(data)
    x2 = self.conv22(x2)

    x3 = self.conv31(data)

    x = torch.cat([x1,x2,x3], dim = 1)
    x = torch.transpose(x, 1,2)
    x = self.fc_final(x)
    x = F.relu(x)

    receivers = torch.bmm(self.rec_idx.repeat(batch_size,1,1), x)
    senders = torch.matmul(self.send_idx, x)
    edges = torch.cat([senders, receivers], dim=2)


    #edge learning
    edges = self.fc1(edges)
    edges = F.relu(edges)
    # edges = F.dropout(edges, p=0.2)

    edges = self.fc2(edges)

    adj = F.gumbel_softmax(edges, tau = 0.5, hard = True)
    adjs = []

    for i in range(batch_size):
        a = adj[i][:, 0].clone().reshape(self.seq_len, -1) #reshape into (seq len, seq len)

        #apply masks
        a = a.masked_fill_(self.tri_mask, 0)
        a = a.masked_fill_(self.diagonal, 0)

        adjs.append(a)

    return torch.stack(adjs, dim = 0)

class Linear(nn.Module):
    r"""An implementation of the linear layer, conducting 2D convolution.
    For details see this paper: `"Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks."
    <https://arxiv.org/pdf/2005.11650.pdf>`_

    Args:
        c_in (int): Number of input channels.
        c_out (int): Number of output channels.
        bias (bool, optional): Whether to have bias. Default: True.
    """

    def __init__(self, c_in: int, c_out: int, bias: bool = True):
        super(Linear, self).__init__()
        self._mlp = torch.nn.Conv2d(
            c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=bias
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, X: torch.FloatTensor) -> torch.FloatTensor:
        """
        Making a forward pass of the linear layer.

        Arg types:
            * **X** (Pytorch Float Tensor) - Input tensor, with shape (batch_size, c_in, num_nodes, seq_len).

        Return types:
            * **X** (PyTorch Float Tensor) - Output tensor, with shape (batch_size, c_out, num_nodes, seq_len).
        """
        return self._mlp(X)


class MixPropModified(nn.Module):
    r"""An implementation of the dynatic mix-hop propagation layer.
    For details see this paper: `"Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks."
    <https://arxiv.org/pdf/2005.11650.pdf>`_

    Args:
        c_in (int): Number of input channels.
        c_out (int): Number of output channels.
        gdep (int): Depth of graph convolution.
        dropout (float): Dropout rate.
        alpha (float): Ratio of retaining the root nodes's original states, a value between 0 and 1.
    """

    def __init__(self, c_in: int, c_out: int, gdep: int, dropout: float, alpha: float, num_nodes:int):
        super(MixPropModified, self).__init__()
        self._mlp = Linear((gdep + 1) * c_in, c_out)
        self._gdep = gdep
        self._dropout = dropout
        self._alpha = alpha

        self.gnns = nn.ModuleList()

        for i in range(self._gdep):
          self.gnns.append(GCNConv(c_in*num_nodes, c_in*num_nodes).to(device))

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, X: torch.FloatTensor, A: torch.FloatTensor) -> torch.FloatTensor:
        """
        Making a forward pass of mix-hop propagation.

        Arg types:
            * **X** (Pytorch Float Tensor) - Input feature Tensor, with shape (batch_size, c_in, num_nodes, seq_len).
            * **A** (PyTorch Float Tensor) - Adjacency matrix, with shape (num_nodes, num_nodes).

        Return types:
            * **H_0** (PyTorch Float Tensor) - Hidden representation for all nodes, with shape (batch_size, c_out, num_nodes, seq_len).
        """
        batch_size, a, b, seq_len = X.shape
        H = X
        H_0 = X
        edge_list, edge_weights = dense_to_sparse(A)

        for i in range(self._gdep):
            # print(X.shape)
            # print(H.shape)

            H = H.reshape([batch_size * seq_len, -1])
            # print(H.shape)
            H1 = self.gnns[i](H,edge_list)
            # print(H1.shape)
            H = self._alpha * X + H1.reshape(batch_size,a,b,seq_len)
            H_0 = torch.cat((H_0, H), dim=1)


        H_0 = self._mlp(H_0)
        return H_0


class DilatedInception(nn.Module):
    r"""An implementation of the dilated inception layer.
    For details see this paper: `"Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks."
    <https://arxiv.org/pdf/2005.11650.pdf>`_

    Args:
        c_in (int): Number of input channels.
        c_out (int): Number of output channels.
        kernel_set (list of int): List of kernel sizes.
        dilated_factor (int, optional): Dilation factor.
    """

    def __init__(self, c_in: int, c_out: int, kernel_set: list, dilation_factor: int):
        super(DilatedInception, self).__init__()
        self._time_conv = nn.ModuleList()
        self._kernel_set = kernel_set
        c_out = int(c_out / len(self._kernel_set))
        for kern in self._kernel_set:
            self._time_conv.append(
                nn.Conv2d(c_in, c_out, (1, kern), dilation=(1, dilation_factor), padding = "same")
            )
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, X_in: torch.FloatTensor) -> torch.FloatTensor:
        """
        Making a forward pass of dilated inception.

        Arg types:
            * **X_in** (Pytorch Float Tensor) - Input feature Tensor, with shape (batch_size, c_in, num_nodes, seq_len).

        Return types:
            * **X** (PyTorch Float Tensor) - Hidden representation for all nodes,
            with shape (batch_size, c_out, num_nodes, seq_len-6).
        """
        X = []
        for i in range(len(self._kernel_set)):
            X.append(self._time_conv[i](X_in))
        for i in range(len(self._kernel_set)):
            X[i] = X[i][..., -X[-1].size(3) :]
        X = torch.cat(X, dim=1)
        return X


class LayerNormalization(nn.Module):
    __constants__ = ["normalized_shape", "weight", "bias", "eps", "elementwise_affine"]
    r"""An implementation of the layer normalization layer.
    For details see this paper: `"Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks."
    <https://arxiv.org/pdf/2005.11650.pdf>`_

    Args:
        normalized_shape (int): Input shape from an expected input of size.
        eps (float, optional): Value added to the denominator for numerical stability. Default: 1e-5.
        elementwise_affine (bool, optional): Whether to conduct elementwise affine transformation or not. Default: True.
    """

    def __init__(
        self, normalized_shape: int, eps: float = 1e-5, elementwise_affine: bool = True
    ):
        super(LayerNormalization, self).__init__()
        self._normalized_shape = tuple(normalized_shape)
        self._eps = eps
        self._elementwise_affine = elementwise_affine
        if self._elementwise_affine:
            self._weight = nn.Parameter(torch.Tensor(*normalized_shape))
            self._bias = nn.Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter("_weight", None)
            self.register_parameter("_bias", None)
        self._reset_parameters()

    def _reset_parameters(self):
        if self._elementwise_affine:
            init.ones_(self._weight)
            init.zeros_(self._bias)

    def forward(self, X: torch.FloatTensor, idx: torch.LongTensor) -> torch.FloatTensor:
        """
        Making a forward pass of layer normalization.

        Arg types:
            * **X** (Pytorch Float Tensor) - Input tensor,
                with shape (batch_size, feature_dim, num_nodes, seq_len).
            * **idx** (Pytorch Long Tensor) - Input indices.

        Return types:
            * **X** (PyTorch Float Tensor) - Output tensor,
                with shape (batch_size, feature_dim, num_nodes, seq_len).
        """
        if self._elementwise_affine:
            return F.layer_norm(
                X,
                tuple(X.shape[1:]),
                self._weight[:, idx, :],
                self._bias[:, idx, :],
                self._eps,
            )
        else:
            return F.layer_norm(
                X, tuple(X.shape[1:]), self._weight, self._bias, self._eps
            )


class MTGNNLayer(nn.Module):
    r"""An implementation of the MTGNN layer.
    For details see this paper: `"Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks."
    <https://arxiv.org/pdf/2005.11650.pdf>`_

    Args:
        dilation_exponential (int): Dilation exponential.
        rf_size_i (int): Size of receptive field.
        kernel_size (int): Size of kernel for convolution, to calculate receptive field size.
        j (int): Iteration index.
        residual_channels (int): Residual channels.
        conv_channels (int): Convolution channels.
        skip_channels (int): Skip channels.
        kernel_set (list of int): List of kernel sizes.
        new_dilation (int): Dilation.
        layer_norm_affline (bool): Whether to do elementwise affine in Layer Normalization.
        gcn_true (bool): Whether to add graph convolution layer.
        seq_length (int): Length of input sequence.
        receptive_field (int): Receptive field.
        dropout (float): Droupout rate.
        gcn_depth (int): Graph convolution depth.
        num_nodes (int): Number of nodes in the graph.
        propalpha (float): Prop alpha, ratio of retaining the root nodes's original states in mix-hop propagation, a value between 0 and 1.

    """

    def __init__(
        self,
        dilation_exponential: int,
        rf_size_i: int,
        kernel_size: int,
        j: int,
        residual_channels: int,
        conv_channels: int,
        skip_channels: int,
        kernel_set: list,
        new_dilation: int,
        layer_norm_affline: bool,
        gcn_true: bool,
        seq_length: int,
        receptive_field: int,
        dropout: float,
        gcn_depth: int,
        num_nodes: int,
        propalpha: float,
    ):
        super(MTGNNLayer, self).__init__()
        self._dropout = dropout
        self._gcn_true = gcn_true

        if dilation_exponential > 1:
            rf_size_j = int(
                rf_size_i
                + (kernel_size - 1)
                * (dilation_exponential ** j - 1)
                / (dilation_exponential - 1)
            )
        else:
            rf_size_j = rf_size_i + j * (kernel_size - 1)

        self._filter_conv = DilatedInception(
            residual_channels,
            conv_channels,
            kernel_set=kernel_set,
            dilation_factor=new_dilation,
        )

        self._gate_conv = DilatedInception(
            residual_channels,
            conv_channels,
            kernel_set=kernel_set,
            dilation_factor=new_dilation,
        )

        self._residual_conv = nn.Conv2d(
            in_channels=conv_channels,
            out_channels=residual_channels,
            kernel_size=(1, 1),
        )

        if seq_length > receptive_field:
            self._skip_conv = nn.Conv2d(
                in_channels=conv_channels,
                out_channels=skip_channels,
                kernel_size=(1, seq_length - rf_size_j + 1),
                padding = "same"
            )
        else:
            self._skip_conv = nn.Conv2d(
                in_channels=conv_channels,
                out_channels=skip_channels,
                kernel_size=(1, receptive_field - rf_size_j + 1),
                padding = "same"
            )

        if gcn_true:
            self._mixprop_conv1 = MixPropModified(
                conv_channels, residual_channels, gcn_depth, dropout, propalpha, num_nodes
            )

            self._mixprop_conv2 = MixPropModified(
                conv_channels, residual_channels, gcn_depth, dropout, propalpha, num_nodes
            )

        if seq_length > receptive_field:
            self._normalization = LayerNormalization(
                (residual_channels, num_nodes, seq_length - rf_size_j + 1),
                elementwise_affine=layer_norm_affline,
            )

        else:
            self._normalization = LayerNormalization(
                (residual_channels, num_nodes, receptive_field - rf_size_j + 1),
                elementwise_affine=layer_norm_affline,
            )
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(
        self,
        X: torch.FloatTensor,
        X_skip: torch.FloatTensor,
        A_tilde: Optional[torch.FloatTensor],
        idx: torch.LongTensor,
        training: bool,
    ) -> torch.FloatTensor:
        """
        Making a forward pass of MTGNN layer.

        Arg types:
            * **X** (PyTorch FloatTensor) - Input feature tensor,
                with shape (batch_size, in_dim, num_nodes, seq_len).
            * **X_skip** (PyTorch FloatTensor) - Input feature tensor for skip connection,
                with shape (batch_size, in_dim, num_nodes, seq_len).
            * **A_tilde** (Pytorch FloatTensor or None) - Predefined adjacency matrix.
            * **idx** (Pytorch LongTensor) - Input indices.
            * **training** (bool) - Whether in traning mode.

        Return types:
            * **X** (PyTorch FloatTensor) - Output sequence tensor,
                with shape (batch_size, seq_len, num_nodes, seq_len).
            * **X_skip** (PyTorch FloatTensor) - Output feature tensor for skip connection,
                with shape (batch_size, in_dim, num_nodes, seq_len).
        """
        X_residual = X
        X_filter = self._filter_conv(X)
        X_filter = torch.tanh(X_filter)
        # print("filter " + str(X_filter.shape))
        X_gate = self._gate_conv(X)
        X_gate = torch.sigmoid(X_gate)
        # print("gate " + str(X_gate.shape))

        X = X_filter * X_gate
        # print("X " + str(X.shape))
        X = F.dropout(X, self._dropout, training=training)
        X_skip1 = self._skip_conv(X) 
        X_skip = X_skip1 + X_skip

        # print("Entering MTGNNLayer GCN block")
        if self._gcn_true:
            # print(A_tilde.shape)
            X1 = self._mixprop_conv1(X, A_tilde)
            X2 = self._mixprop_conv2(X, A_tilde.transpose(2, 1))
            X = X1 + X2
        else:
            X = self._residual_conv(X)

        X = X + X_residual[:, :, :, -X.size(3) :]
        X = self._normalization(X, idx)
        return X, X_skip


class TimeMTGNN(nn.Module):
    r"""An implementation of the Multivariate Time Series Forecasting Graph Neural Networks.
    For details see this paper: `"Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks."
    <https://arxiv.org/pdf/2005.11650.pdf>`_

    Args:
        gcn_true (bool): Whether to add graph convolution layer.
        build_adj (bool): Whether to construct adaptive adjacency matrix.
        gcn_depth (int): Graph convolution depth.
        num_nodes (int): Number of nodes in the graph.
        kernel_set (list of int): List of kernel sizes.
        kernel_size (int): Size of kernel for convolution, to calculate receptive field size.
        dropout (float): Droupout rate.
        subgraph_size (int): Size of subgraph.
        node_dim (int): Dimension of nodes.
        dilation_exponential (int): Dilation exponential.
        conv_channels (int): Convolution channels.
        residual_channels (int): Residual channels.
        skip_channels (int): Skip channels.
        end_channels (int): End channels.
        seq_length (int): Length of input sequence.
        in_dim (int): Input dimension.
        out_dim (int): Output dimension.
        layers (int): Number of layers.
        propalpha (float): Prop alpha, ratio of retaining the root nodes's original states in mix-hop propagation, a value between 0 and 1.
        tanhalpha (float): Tanh alpha for generating adjacency matrix, alpha controls the saturation rate.
        layer_norm_affline (bool): Whether to do elementwise affine in Layer Normalization.
        xd (int, optional): Static feature dimension, default None.
    """

    def __init__(
        self,
        gcn_true: bool,
        build_adj: bool,
        gcn_depth: int,
        num_nodes: int,
        kernel_set: list,
        kernel_size: int,
        dropout: float,
        subgraph_size: int,
        node_dim: int,
        dilation_exponential: int,
        conv_channels: int,
        residual_channels: int,
        skip_channels: int,
        end_channels: int,
        seq_length: int,
        in_dim: int,
        out_dim: int,
        layers: int,
        propalpha: float,
        tanhalpha: float,
        layer_norm_affline: bool,
        xd: Optional[int] = None,
    ):
        super(TimeMTGNN, self).__init__()

        self._gcn_true = gcn_true
        self._build_adj_true = build_adj
        self._num_nodes = num_nodes
        self._dropout = dropout
        self._seq_length = seq_length
        self._layers = layers
        self._idx = torch.arange(self._num_nodes)
        self._graph_constructor = TimeGraphConstructorMTGNN(num_nodes, residual_channels, seq_length)

        self._mtgnn_layers = nn.ModuleList()

        self._set_receptive_field(dilation_exponential, kernel_size, layers)

        new_dilation = 1
        for j in range(1, layers + 1):
            self._mtgnn_layers.append(
                MTGNNLayer(
                    dilation_exponential=dilation_exponential,
                    rf_size_i=1,
                    kernel_size=kernel_size,
                    j=j,
                    residual_channels=residual_channels,
                    conv_channels=conv_channels,
                    skip_channels=skip_channels,
                    kernel_set=kernel_set,
                    new_dilation=new_dilation,
                    layer_norm_affline=layer_norm_affline,
                    gcn_true=gcn_true,
                    seq_length=seq_length,
                    receptive_field=self._receptive_field,
                    dropout=dropout,
                    gcn_depth=gcn_depth,
                    num_nodes=num_nodes,
                    propalpha=propalpha,
                )
            )

            new_dilation *= dilation_exponential

        self._setup_conv(
            in_dim, skip_channels, end_channels, residual_channels, out_dim
        )

        self.final_FC = nn.Linear(self._seq_length, 1)

        self._reset_parameters()

    def _setup_conv(
        self, in_dim, skip_channels, end_channels, residual_channels, out_dim
    ):

        # print(self._receptive_field)
        self._start_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=residual_channels, kernel_size=(1, 1), padding = "same"
        )

        if self._seq_length > self._receptive_field:

            self._skip_conv_0 = nn.Conv2d(
                in_channels=in_dim,
                out_channels=skip_channels,
                kernel_size=(1, self._seq_length),
                bias=True,
            )

            self._skip_conv_E = nn.Conv2d(
                in_channels=residual_channels,
                out_channels=skip_channels,
                kernel_size=(1, self._seq_length - self._receptive_field + 1),
                bias=True,
                padding = "same"
            )

        else:
            self._skip_conv_0 = nn.Conv2d(
                in_channels=in_dim,
                out_channels=skip_channels,
                kernel_size=(1, self._receptive_field),
                bias=True,
                padding = "same"
            )

            self._skip_conv_E = nn.Conv2d(
                in_channels=residual_channels,
                out_channels=skip_channels,
                kernel_size=(1, 1),
                bias=True,
                padding = "same"
            )

        self._end_conv_1 = nn.Conv2d(
            in_channels=skip_channels,
            out_channels=end_channels,
            kernel_size=(1, 1),
            bias=True,
        )

        self._end_conv_2 = nn.Conv2d(
            in_channels=end_channels,
            out_channels=out_dim,
            kernel_size=(1, 1),
            bias=True,
        )

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def _set_receptive_field(self, dilation_exponential, kernel_size, layers):
        if dilation_exponential > 1:
            self._receptive_field = int(
                1
                + (kernel_size - 1)
                * (dilation_exponential ** layers - 1)
                / (dilation_exponential - 1)
            )
        else:
            self._receptive_field = layers * (kernel_size - 1) + 1

    def forward(
        self,
        X_in: torch.FloatTensor,
        A_tilde: Optional[torch.FloatTensor] = None,
        idx: Optional[torch.LongTensor] = None,
        FE: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        """
        Making a forward pass of MTGNN.

        Arg types:
            * **X_in** (PyTorch FloatTensor) - Input sequence, with shape (batch_size, in_dim, num_nodes, seq_len).
            * **A_tilde** (Pytorch FloatTensor, optional) - Predefined adjacency matrix, default None.
            * **idx** (Pytorch LongTensor, optional) - Input indices, a permutation of the num_nodes, default None (no permutation).
            * **FE** (Pytorch FloatTensor, optional) - Static feature, default None.

        Return types:
            * **X** (PyTorch FloatTensor) - Output sequence for prediction, with shape (batch_size, seq_len, num_nodes, 1).
        """
        seq_len = X_in.size(3)
        X = X_in
        assert (
            seq_len == self._seq_length
        ), "Input sequence length not equal to preset sequence length."

        # if self._seq_length < self._receptive_field:
        #     X_in = nn.functional.pad(
        #         X_in, (self._receptive_field - self._seq_length, 0, 0, 0)
        #     )

        if self._gcn_true:
            if self._build_adj_true:
                A_tilde = self._graph_constructor(X_in)

        # print("past graph constructor")
        X = self._start_conv(X_in)
        # print(X.shape)
        X_skip = self._skip_conv_0(
            F.dropout(X_in, self._dropout, training=self.training)
        )
        # print("past initial convs")
        if idx is None:
            for mtgnn in self._mtgnn_layers:
                X, X_skip = mtgnn(
                    X, X_skip, A_tilde, self._idx.to(X_in.device), self.training
                )
        else:
            for mtgnn in self._mtgnn_layers:
                X, X_skip = mtgnn(X, X_skip, A_tilde, idx, self.training)
        # print("past mtgnn layer block")
        
        X_skip = self._skip_conv_E(X)  + X_skip
        X = F.relu(X_skip)
        X = F.relu(self._end_conv_1(X))
        X = self._end_conv_2(X)
        X = self.final_FC(X)
        return X