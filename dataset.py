
import os
import sys
import scipy.io
from typing import Optional, Any
import numpy as np
from torch_geometric.data import Data, Dataset
import torch_geometric.transforms
import torch
from torchvision.datasets.folder import default_loader
from torchvision import transforms


class MMGraphDataset(Dataset):
    """ Dataset that loads graph data on demand."""

    def __init__(self,
                 cell_graph_path: Optional[str] = None,
                 patch_graph_lv0_path: Optional[str] = None,
                 patch_graph_lv1_path: Optional[str] = None,
                 tissue_graph_path: Optional[str] = None,
                 use_cell_graph: bool = True,
                 use_patch_graph_lv0: bool = True,
                 use_patch_graph_lv1: bool = True,
                 use_tissue_graph: bool = True,
                 gnn_transform: Any = None,
                 train_mode: bool = True,
                 normalize_features: bool = True):
        super().__init__()
        self.use_cell_graph = use_cell_graph
        self.use_patch_graph_lv0 = use_patch_graph_lv0
        self.use_patch_graph_lv1 = use_patch_graph_lv1
        self.use_tissue_graph = use_tissue_graph
        self.normalize_features = normalize_features

        self.cell_graph_path = cell_graph_path if use_cell_graph else None
        self.patch_graph_lv0_path = patch_graph_lv0_path if use_patch_graph_lv0 else None
        self.patch_graph_lv1_path = patch_graph_lv1_path if use_patch_graph_lv1 else None
        self.tissue_graph_path = tissue_graph_path if use_tissue_graph else None

        if gnn_transform or not train_mode:
            self.gnn_transform = gnn_transform
        elif train_mode:
            self.gnn_transform = torch_geometric.transforms.Compose([
                torch_geometric.transforms.RandomJitter(0.01),
            ])

        # Collect filenames from the first available graph path
        paths = [self.cell_graph_path, self.patch_graph_lv0_path, self.patch_graph_lv1_path, self.tissue_graph_path]
        active_path = next((p for p in paths if p is not None), None)

        # 取一个非空地址得到文件名称
        if active_path:
            self.graph_filenames = sorted([f for f in os.listdir(active_path) if f.endswith('.pt')])
        else:
            self.graph_filenames = []



    def __len__(self):
        return len(self.graph_filenames)

    def __getitem__(self, index):
        # Get the filename for the current index
        filename = self.graph_filenames[index]
        data = {}
        # Load graphs based on their availability and required paths
        if self.use_cell_graph:
            cell_file = os.path.join(self.cell_graph_path, filename)
            cell_graph = torch.load(cell_file)
            if self.normalize_features:
                min_val = cell_graph.x.min(dim=0, keepdim=True)[0]
                max_val = cell_graph.x.max(dim=0, keepdim=True)[0]
                cell_graph.x = 2 * ((cell_graph.x - min_val) / (max_val - min_val + 1e-6)) - 1
            if self.gnn_transform:
                cell_graph = self.gnn_transform(cell_graph)
            data['cell_graph'] = cell_graph

        if self.use_patch_graph_lv0:
            patch_lv0_file = os.path.join(self.patch_graph_lv0_path, filename)
            patch_graph_lv0 = torch.load(patch_lv0_file)
            if self.normalize_features:
                min_val = patch_graph_lv0.x.min(dim=0, keepdim=True)[0]
                max_val = patch_graph_lv0.x.max(dim=0, keepdim=True)[0]
                patch_graph_lv0.x = 2 * ((patch_graph_lv0.x - min_val) / (max_val - min_val + 1e-6)) - 1
            if self.gnn_transform:
                patch_graph_lv0 = self.gnn_transform(patch_graph_lv0)
            data['patch_graph_lv0'] = patch_graph_lv0

        if self.use_patch_graph_lv1:
            patch_lv1_file = os.path.join(self.patch_graph_lv1_path, filename)
            patch_graph_lv1 = torch.load(patch_lv1_file)
            if self.normalize_features:
                min_val = patch_graph_lv1.x.min(dim=0, keepdim=True)[0]
                max_val = patch_graph_lv1.x.max(dim=0, keepdim=True)[0]
                patch_graph_lv1.x = 2 * ((patch_graph_lv1.x - min_val) / (max_val - min_val + 1e-6)) - 1
            if self.gnn_transform:
                patch_graph_lv1 = self.gnn_transform(patch_graph_lv1)
            data['patch_graph_lv1'] = patch_graph_lv1

        if self.use_tissue_graph:
            tissue_file = os.path.join(self.tissue_graph_path, filename)
            tissue_graph = torch.load(tissue_file)
            if self.normalize_features:
                min_val = tissue_graph.x.min(dim=0, keepdim=True)[0]
                max_val = tissue_graph.x.max(dim=0, keepdim=True)[0]
                tissue_graph.x = 2 * ((tissue_graph.x - min_val) / (max_val - min_val + 1e-6)) - 1
            if self.gnn_transform:
                tissue_graph = self.gnn_transform(tissue_graph)
            data['tissue_graph'] = tissue_graph

        # Verify that all loaded graphs have the same label
        labels = [graph.y for graph in data.values() if hasattr(graph, 'y')]
        if all(torch.equal(labels[0], label) for label in labels):
            label = labels[0].squeeze()
        else:
            raise ValueError("Not all graphs have the same label.")

        return data, label

    @property
    def num_features(self):
        # Initialize dictionary to store the number of features for each graph type
        features = {}

        # Load the first available graph for each type and retrieve the number of features
        if self.use_cell_graph and self.cell_graph_path:
            first_cell_graph = torch.load(os.path.join(self.cell_graph_path, self.graph_filenames[0]))
            features['cell_graph'] = first_cell_graph.x.shape[1]

        if self.use_patch_graph_lv0 and self.patch_graph_lv0_path:
            first_patch_graph_lv0 = torch.load(os.path.join(self.patch_graph_lv0_path, self.graph_filenames[0]))
            features['patch_graph_lv0'] = first_patch_graph_lv0.x.shape[1]

        if self.use_patch_graph_lv1 and self.patch_graph_lv1_path:
            first_patch_graph_lv1 = torch.load(os.path.join(self.patch_graph_lv1_path, self.graph_filenames[0]))
            features['patch_graph_lv1'] = first_patch_graph_lv1.x.shape[1]

        if self.use_tissue_graph and self.tissue_graph_path:
            first_tissue_graph = torch.load(os.path.join(self.tissue_graph_path, self.graph_filenames[0]))
            features['tissue_graph'] = first_tissue_graph.x.shape[1]

        return features
