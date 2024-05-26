# Taken from EGNN repo with minor changes made

import numpy as np
import torch
from torch.utils import data


def get_edges(adjacency_matrices):
    batch_size, n_nodes, _ = adjacency_matrices.shape

    # Generate indices for all possible pairs of nodes
    rows, cols = np.meshgrid(np.arange(n_nodes), np.arange(n_nodes), indexing='ij')

    # Flatten the indices arrays and filter out self-loops
    mask = rows != cols
    rows = rows[mask]
    cols = cols[mask]

    # Gather edge attributes
    edge_attr = adjacency_matrices[:, rows, cols]

    # Initialize the result tensor
    edges_tensor = np.stack((rows, cols))

    # Prepare the tensor with PyTorch
    edge_attr_tensor = torch.from_numpy(edge_attr).unsqueeze(2).float()
    edges_tensor = torch.tensor(edges_tensor, dtype=torch.int64)

    return edges_tensor, edge_attr_tensor


class NBodyDataset:
    def __init__(self, partition, data_root ='nbody_dataset/', suffix='_charged5_initvel1small', max_samples=1000):

        self.suffix = suffix  # '_charged5_initvel1small'
        self.data_root = data_root  # 'nbody_dataset/'
        self.max_samples = int(max_samples)
        self.partition = partition  # train, val, test

        self.data, self.edges = self.load()

    def load(self):
       # Load the .npy files for loc, vel, edges, and charges
        loc = np.load(self.data_root + "loc_" + self.partition + self.suffix + ".npy")
        vel = np.load(self.data_root + "vel_" + self.partition + self.suffix + ".npy")
        edges = np.load(self.data_root + "edges_" + self.partition + self.suffix + ".npy")
        charges = np.load(self.data_root + "charges_" + self.partition + self.suffix + ".npy")
        # Preprocess the data
        loc, vel, edge_attr, edges, charges = self.preprocess(loc, vel, edges, charges)

        return (loc, vel, edge_attr, charges), edges

    def preprocess(self, loc, vel, edges, charges):
        #print("before",loc.shape, vel.shape, edges.shape, charges.shape)
        # Convert arrays to tensors and adjust dimension ordering
        loc = torch.tensor(loc).float().permute(0, 1, 3, 2)  # [batch, nodes, features, time_steps]
        vel = torch.tensor(vel).float().permute(0, 1, 3, 2)  # [batch, nodes, features, time_steps]
        charges = torch.tensor(charges).float()
        #print("after", loc.shape, vel.shape, edges.shape, charges.shape)

        # Limit the number of samples if max_samples is set
        if self.max_samples is not None:
            loc, vel, edges, charges = self.limit_samples(loc, vel, edges, charges)

        # Handle edges
        edges, edge_attr = get_edges(edges)
        return loc, vel, edge_attr, edges, charges

    def limit_samples(self, loc, vel, edges, charges):
        min_size = min(loc.size(0), self.max_samples)
        loc = loc[:min_size]
        vel = vel[:min_size]
        charges = charges[:min_size]
        edges = edges[:min_size]
        return loc, vel, edges, charges

    def __getitem__(self, i):
        loc, vel, edge_attr, charges = self.data
        loc, vel, edge_attr, charges = loc[i], vel[i], edge_attr[i], charges[i]

        if self.suffix == '_charged5_initvel1small':
            frame_0, frame_T = 30, 40
        else:
            raise Exception("Wrong dataset partition %s" % self.suffix)

        return loc[frame_0], vel[frame_0], edge_attr, charges, loc[frame_T], self.edges

    def __len__(self):
        return len(self.data[0])

    def get_n_nodes(self):
        return self.data[0].size(1)


class NBody:
    def __init__(self, data_root ='nbody_dataset/', num_samples=3000, batch_size=100):
        self.train_dataset = NBodyDataset(
            partition="train", data_root=data_root, max_samples=num_samples, suffix='_charged5_initvel1small'
        )
        self.valid_dataset = NBodyDataset(
            partition="valid", data_root=data_root, max_samples=num_samples, suffix='_charged5_initvel1small'
        )
        self.test_dataset = NBodyDataset(
            partition="test", data_root=data_root, max_samples=num_samples, suffix='_charged5_initvel1small'
        )

        self.batch_size = batch_size

    def train_loader(self):
        return data.DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True
        )

    def val_loader(self):
        return data.DataLoader(
            self.valid_dataset, batch_size=self.batch_size, shuffle=False
        )

    def test_loader(self):
        return data.DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False
        )