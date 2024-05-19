import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model.nbody_transformer import NBodyTransformer
from algebra.cliffordalgebra import CliffordAlgebra
from data.nbody import NBody
from torch.optim.lr_scheduler import CosineAnnealingLR


def train_epoch(model, train_loader, criterion, optimizer, scheduler):
    model.train()
    running_loss = 0.0
    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()
        output, tgt = model(batch)
        loss = criterion(output, tgt)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()
        scheduler.step()  # Update the learning rate per step
        running_loss += loss.item()
    return running_loss / len(train_loader)


def validate_epoch(model, val_loader, criterion):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            output, tgt = model(batch)
            loss = criterion(output, tgt)
            running_loss += loss.item()
    return running_loss / len(val_loader)


metric = [1, 1, 1]
clifford_algebra = CliffordAlgebra(metric)

# Hyperparameters
input_dim = 3  # feature_dim
d_model = 16
num_heads = 8
num_layers = 4

batch_size = 100
num_samples = 3000

# Create the model
model = NBodyTransformer(input_dim, d_model, num_heads, num_layers, clifford_algebra, unique_edges=True)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=5e-3, weight_decay=0.0001)

nbody_data = NBody(num_samples=num_samples, batch_size=batch_size)

train_loader = nbody_data.train_loader()
val_loader = nbody_data.val_loader()  # Assuming you have a validation data loader
epochs = 1000

steps_per_epoch = len(train_loader)  # number of batches per epoch
steps = epochs * steps_per_epoch

scheduler = CosineAnnealingLR(
    optimizer,
    T_max=steps
)

best_val_loss = float('inf')
early_stopping_counter = 0
early_stopping_limit = 50

for epoch in tqdm(range(epochs)):
    train_loss = train_epoch(model, train_loader, criterion, optimizer, scheduler)
    val_loss = validate_epoch(model, val_loader, criterion)

    # Save model if validation loss improved
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pth')
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1

    # Stop training if validation loss has not improved for early_stopping_limit epochs
    if early_stopping_counter >= early_stopping_limit:
        print('Early stopping...')
        break

    print(f'Epoch {epoch + 1}, Training Loss: {train_loss}, Validation Loss: {val_loss}')
