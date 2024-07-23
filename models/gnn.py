from multiprocessing import pool
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin
from torch_geometric.nn import (
    GraphConv, 
    GINConv,
    GATConv,
    TopKPooling, 
    BatchNorm,
    # LayerNorm,
)
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from typing import Optional

ALL_ACT_LAYERS = {
    "leaky_relu": nn.LeakyReLU,
    "gelu": nn.GELU,
    "relu": nn.ReLU,
}


class GCNTopK2(nn.Module):
    def __init__(
            self,
            in_features: int,
            hidden_features: Optional[int] = 256,
            out_features: Optional[int] = 256,
            pooling: Optional[float] = 0.5,
            activation: Optional[str] = "gelu",
    ):
        super(GCNTopK2, self).__init__()

        self.activation = ALL_ACT_LAYERS[activation]()

        self.conv1 = GraphConv(in_features, hidden_features)
        self.bn1 = BatchNorm(hidden_features)
        self.pool1 = TopKPooling(hidden_features, ratio=pooling)

        self.conv2 = GraphConv(hidden_features, hidden_features)
        self.pool2 = TopKPooling(hidden_features, ratio=pooling)
        self.bn2 = BatchNorm(hidden_features)

        self.linear = torch.nn.Linear(hidden_features * 2, out_features)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = self.conv1(x, edge_index)
        x = self.activation(x)
        x = self.bn1(x)

        x, edge_index, edge_attr, batch, _, _ = self.pool1(x, edge_index, edge_attr, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = self.bn2(x)
        x = self.activation(x)
        x = self.conv2(x, edge_index)

        x, edge_index, edge_attr, batch, _, _ = self.pool2(x, edge_index, edge_attr, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2
        x = self.linear(x)

        return x


class GINTopK2(nn.Module):
    # def __init__(self, num_feature, num_class, nhid):
    def __init__(
            self,
            in_features: int,
            hidden_features: Optional[int] = 256,
            out_features: Optional[int] = 256,
            pooling: Optional[float] = 0.5,
            activation: Optional[str] = "gelu",
    ):
        super(GINTopK2, self).__init__()

        self.activation = ALL_ACT_LAYERS[activation]()

        self.conv1 = GINConv(
            Seq(
                Lin(in_features, hidden_features),
                self.activation,
                Lin(hidden_features, hidden_features)
            )
        )

        self.bn1 = BatchNorm(hidden_features)
        self.pool1 = TopKPooling(hidden_features, ratio=pooling)

        self.conv2 = GINConv(
            Seq(
                Lin(hidden_features, hidden_features),
                self.activation,
                Lin(hidden_features, hidden_features)
            )
        )

        self.bn2 = BatchNorm(hidden_features)
        self.pool2 = TopKPooling(hidden_features, ratio=pooling)

        self.linear = torch.nn.Linear(hidden_features * 2, out_features)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = self.activation(x)
        x = self.bn1(x)

        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = self.conv2(x, edge_index)
        x = self.activation(x)
        x = self.bn2(x)

        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2

        x = self.linear(x)

        return x


class GATTopK2(nn.Module):
    def __init__(
            self,
            in_features: int,
            hidden_features: Optional[int] = 256,
            out_features: Optional[int] = 256,
            pooling: Optional[float] = 0.5,
            activation: Optional[str] = "gelu",
            heads: Optional[int] = 4,
    ):
        super(GATTopK2, self).__init__()

        self.activation = ALL_ACT_LAYERS[activation]()

        self.conv1 = GATConv(in_features, hidden_features, heads=heads)
        self.bn1 = BatchNorm(hidden_features * heads)
        self.pool1 = TopKPooling(hidden_features * heads, ratio=pooling)

        self.conv2 = GATConv(hidden_features * heads, hidden_features, heads=heads)
        self.bn2 = BatchNorm(hidden_features * heads)
        self.pool2 = TopKPooling(hidden_features * heads, ratio=pooling)

        self.linear = torch.nn.Linear(hidden_features * heads * 2, out_features)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = self.activation(x)
        x = self.bn1(x)

        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = self.conv2(x, edge_index)
        x = self.activation(x)
        x = self.bn2(x)

        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2
        x = self.linear(x)

        return x





class GCNTopK4(nn.Module):
    def __init__(
            self,
            in_features: int,
            hidden_features: Optional[int] = 256,
            out_features: Optional[int] = 256,
            pooling: Optional[float] = 0.5,
            activation: Optional[str] = "gelu",
    ):
        super(GCNTopK4, self).__init__()

        self.activation = ALL_ACT_LAYERS[activation]()

        self.conv1 = GraphConv(int(in_features), hidden_features)
        self.bn1 = BatchNorm(hidden_features)
        self.pool1 = TopKPooling(hidden_features, ratio=pooling)

        self.conv2 = GraphConv(hidden_features, hidden_features)
        self.pool2 = TopKPooling(hidden_features, ratio=pooling)
        self.bn2 = BatchNorm(hidden_features)

        self.conv3 = GraphConv(hidden_features, hidden_features)
        self.pool3 = TopKPooling(hidden_features, ratio=pooling)
        self.bn3 = BatchNorm(hidden_features)

        # add one more conv-pooling block, i.e., conv4 and pool4
        self.conv4 = GraphConv(hidden_features, hidden_features)
        self.pool4 = TopKPooling(hidden_features, ratio=pooling)
        self.bn4 = BatchNorm(hidden_features)

        self.linear = torch.nn.Linear(hidden_features * 2, out_features)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = self.conv1(x, edge_index)
        x = self.activation(x)
        x = self.bn1(x)

        x, edge_index, edge_attr, batch, _, _ = self.pool1(x, edge_index, edge_attr, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = self.conv2(x, edge_index)
        x = self.activation(x)
        x = self.bn2(x)

        x, edge_index, edge_attr, batch, _, _ = self.pool2(x, edge_index, edge_attr, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = self.conv3(x, edge_index)
        x = self.activation(x)
        x = self.bn3(x)

        x, edge_index, edge_attr, batch, _, _ = self.pool3(x, edge_index, edge_attr, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = self.conv4(x, edge_index)
        x = self.activation(x)
        x = self.bn4(x)

        x, edge_index, edge_attr, batch, _, _ = self.pool4(x, edge_index, edge_attr, batch)
        x4 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3 + x4

        x = self.linear(x)

        return x



class GINTopK4(nn.Module):
    def __init__(
            self,
            in_features: int,
            hidden_features: Optional[int] = 256,
            out_features: Optional[int] = 256,
            pooling: Optional[float] = 0.5,
            activation: Optional[str] = "gelu",
    ):
        super(GINTopK4, self).__init__()

        self.activation_fn = ALL_ACT_LAYERS[activation]()

        self.conv1 = GINConv(
            Seq(Lin(in_features, hidden_features), self.activation_fn, Lin(hidden_features, hidden_features)))
        self.bn1 = BatchNorm(hidden_features)
        self.pool1 = TopKPooling(hidden_features, ratio=pooling)

        self.conv2 = GINConv(
            Seq(Lin(hidden_features, hidden_features), self.activation_fn, Lin(hidden_features, hidden_features)))
        self.bn2 = BatchNorm(hidden_features)
        self.pool2 = TopKPooling(hidden_features, ratio=pooling)

        self.conv3 = GINConv(
            Seq(Lin(hidden_features, hidden_features), self.activation_fn, Lin(hidden_features, hidden_features)))
        self.bn3 = BatchNorm(hidden_features)
        self.pool3 = TopKPooling(hidden_features, ratio=pooling)

        self.conv4 = GINConv(
            Seq(Lin(hidden_features, hidden_features), self.activation_fn, Lin(hidden_features, hidden_features)))
        self.bn4 = BatchNorm(hidden_features)
        self.pool4 = TopKPooling(hidden_features, ratio=pooling)

        # Linear transformation to the final output features
        self.linear = nn.Linear(hidden_features * 2, out_features)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = self.activation_fn(x)
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = self.activation_fn(x)
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = self.activation_fn(x)
        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = self.conv4(x, edge_index)
        x = self.bn4(x)
        x = self.activation_fn(x)
        x, edge_index, _, batch, _, _ = self.pool4(x, edge_index, None, batch)
        x4 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3 + x4
        x = self.linear(x)

        return x





    


class GATTopK4(nn.Module):
    def __init__(
            self,
            in_features: int,
            hidden_features: Optional[int] = 256,
            out_features: Optional[int] = 256,
            pooling: Optional[float] = 0.5,
            activation: Optional[str] = "gelu",
            heads: Optional[int] = 4,
    ):
        super(GATTopK4, self).__init__()

        self.activation = ALL_ACT_LAYERS[activation]()

        self.conv1 = GATConv(in_features, hidden_features, heads=heads)
        self.bn1 = BatchNorm(hidden_features * heads)
        self.pool1 = TopKPooling(hidden_features * heads, ratio=pooling)

        self.conv2 = GATConv(hidden_features * heads, hidden_features, heads=heads)
        self.bn2 = BatchNorm(hidden_features * heads)
        self.pool2 = TopKPooling(hidden_features * heads, ratio=pooling)

        self.conv3 = GATConv(hidden_features * heads, hidden_features, heads=heads)
        self.bn3 = BatchNorm(hidden_features * heads)
        self.pool3 = TopKPooling(hidden_features * heads, ratio=pooling)

        self.conv4 = GATConv(hidden_features * heads, hidden_features, heads=heads)
        self.bn4 = BatchNorm(hidden_features * heads)
        self.pool4 = TopKPooling(hidden_features * heads, ratio=pooling)

        self.linear = torch.nn.Linear(hidden_features * heads * 2, out_features)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = self.activation(x)
        x = self.bn1(x)

        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = self.conv2(x, edge_index)
        x = self.activation(x)
        x = self.bn2(x)

        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = self.conv3(x, edge_index)
        x = self.activation(x)
        x = self.bn3(x)

        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = self.conv4(x, edge_index)
        x = self.activation(x)
        x = self.bn4(x)

        x, edge_index, _, batch, _, _ = self.pool4(x, edge_index, None, batch)
        x4 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3 + x4
        x = self.linear(x)

        return x




if __name__ == '__main__':
    model = GCNTopK2(in_features=223)
    