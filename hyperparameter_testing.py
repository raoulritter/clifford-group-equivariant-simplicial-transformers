import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model.nbody_transformer import NBodyTransformer
from algebra.cliffordalgebra import CliffordAlgebra
from dataset import NBody
import optuna
from torch.optim.lr_scheduler import CosineAnnealingLR
import joblib


def objective(trial):
    # Define search space for hyperparameters
    input_dim = 3
    d_model = trial.suggest_categorical('d_model', [16, 32, 64, 128])
    num_heads = trial.suggest_categorical('num_heads', [4, 8, 16])
    num_layers = trial.suggest_int('num_layers', 1, 8)
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [50, 100, 150, 200])
    wd = trial.suggest_float('wd', 1e-6, 1e-2, log=True)


    clifford_algebra = CliffordAlgebra([1, 1, 1])

    num_samples = 500

    # Create the model
    model = NBodyTransformer(input_dim, d_model, num_heads, num_layers, clifford_algebra, unique_edges=True)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    nbody_data = NBody(num_samples=num_samples, batch_size=batch_size)
    train_loader = nbody_data.train_loader()
    val_loader = nbody_data.val_loader()
    steps_per_epoch = len(train_loader)  # number of batches per epoch
    steps = 50 * steps_per_epoch

    scheduler = CosineAnnealingLR(
        optimizer,
        steps * 50
    )

    nbody_data = NBody(num_samples=num_samples, batch_size=batch_size)
    train_loader = nbody_data.train_loader()
    val_loader = nbody_data.val_loader()

    def train_epoch(model, train_loader, criterion, optimizer):
        model.train()
        running_loss = 0.0
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            output, tgt = model(batch)
            loss = criterion(output, tgt)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
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

    epochs = 50
    for epoch in tqdm(range(epochs)):
        train_loss = train_epoch(model, train_loader, criterion, optimizer)
        val_loss = validate_epoch(model, val_loader, criterion)
        scheduler.step()

        trial.report(val_loss, epoch)

        # Handle pruning based on the intermediate value
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return val_loss


if __name__ == "__main__":
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=1000)

    print("Best hyperparameters: ", study.best_params)
    print("Best validation loss: ", study.best_value)

    # save study to file
    study_name = "nbody_study"
    joblib.dump(study, "study.pkl")

