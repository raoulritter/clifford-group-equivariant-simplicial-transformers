import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from data_utils import embed_nbody_graphs
from models.nbody_transformer import NBODY_Transformer
from algebra.cliffordalgebra import CliffordAlgebra
from torch.utils.tensorboard import SummaryWriter
from data.nbody import NBody

# Define the metric for 3D space (Euclidean)
metric = [1, 1, 1]
clifford_algebra = CliffordAlgebra(metric)

# Hyperparameters
input_dim = 7  # feature_dim
d_model = 7 # hidden_dim for transformer
num_heads = 8 # number of heads in transformer
num_layers = 6 # number of transformer layers
batch_size = 100
channels = 7   #????????
num_samples = 3000

# Create the model
model = NBODY_Transformer(input_dim, d_model, num_heads, num_layers, batch_size, clifford_algebra, channels)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

nbody_data = NBody(num_samples=num_samples, batch_size=batch_size)
train_loader = nbody_data.train_loader()
# val_loader = nbody_data.val_loader()
model.train()

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad()
        components = embed_nbody_graphs(batch)
        nodes, edges, tgt, attention_mask = components

        output = model(nodes, edges, src_mask=attention_mask, batch_size=batch_size)
        loss = criterion(output, tgt)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


for epoch in tqdm(range(100)):
    train_loss = train_epoch(model, train_loader, criterion, optimizer)
    print(f"Epoch {epoch}: Train Loss {train_loss}")