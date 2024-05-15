import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from data_utils import generate_mock_batch, embed_nbody_graphs
from models.nbody_transformer import NBODY_Transformer
from algebra.cliffordalgebra import CliffordAlgebra
from data.nbody import NBody

# Define the metric for 3D space (Euclidean)
metric = [1, 1, 1]
clifford_algebra = CliffordAlgebra(metric)

# Hyperparameters
input_dim = 7  # feature_dim
d_model = 7
num_heads = 8
num_layers = 6
embed_in_features = 3
embed_out_features = 3
batch_size = 1
channels = 7
num_samples = 256

# Create the model
model = NBODY_Transformer(input_dim, d_model, num_heads, num_layers, batch_size, embed_in_features, embed_out_features, clifford_algebra, channels)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

demo_batch = generate_mock_batch(batch_size)
nbody_data = NBody(num_samples=num_samples, batch_size=batch_size)
train_loader = nbody_data.train_loader()
print("Data loaded")
src_mask = None

model.train()
for epoch in tqdm(range(10)):
    for batch in train_loader:
        optimizer.zero_grad()
        components = embed_nbody_graphs(batch)
        nodes, edges, tgt, attention_mask = components

        output = model(nodes, edges, src_mask=attention_mask, batch_size=batch_size)

        loss = criterion(output, tgt)
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch + 1}, Loss: {loss.item()}')
