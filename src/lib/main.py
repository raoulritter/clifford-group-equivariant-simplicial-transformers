import torch
import torch.nn as nn
import torch.optim as optim
from nbody_model.modules.transformer import NBodyTransformer
from nbody_model.algebra import CliffordAlgebra
from nbody_model.data.nbody import NBody
# from .data.nbody import NBody
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse
import csv


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
        scheduler.step()
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


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train and validate NBodyTransformer nbody_model.")
    parser.add_argument('--d_model', type=int, default=128, help='Dimension of the nbody_model')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=5, help='Number of layers')
    parser.add_argument('--lr', type=float, default=0.000248, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=50, help='Batch size')
    parser.add_argument('--num_samples', type=int, default=3000, help='Number of samples')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--num_edges', type=int, choices=[0, 10, 20], default=10, help='Number of edges')
    parser.add_argument('--weight_decay', type=float, default=0.00001, help='Weight decay')
    parser.add_argument('--early_stopping_limit', type=int, default=50, help='Early stopping limit')
    parser.add_argument('--zero_edges', action='store_true', help='Flag to indicate zero edges')
    parser.add_argument('--test_only', action='store_true', help='Flag to indicate find test loss')
    return parser.parse_args()


def save_losses_to_csv(args, train_losses, val_losses, test_loss, filename='losses.csv'):
    filename = f'../../results/{args.num_edges}_{args.zero_edges}_{filename}'
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Hyperparameters'])
        for key, value in vars(args).items():
            writer.writerow([key, value])
        writer.writerow(['Epoch', 'Training Loss', 'Validation Loss'])
        for epoch, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses), start=1):
            writer.writerow([epoch, train_loss, val_loss])
        writer.writerow(['Test Loss'])
        writer.writerow([test_loss])


def test_model(model, test_loader, criterion):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            output, tgt = model(batch)
            loss = criterion(output, tgt)
            running_loss += loss.item()
    return running_loss / len(test_loader)


def main():
    args = parse_arguments()
    if args.test_only:
        model = NBodyTransformer(
            input_dim=3,
            d_model=args.d_model,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            clifford_algebra=CliffordAlgebra([1, 1, 1]),
            num_edges=args.num_edges,
            zero_edges=args.zero_edges
        )
        model.load_state_dict(torch.load(f'../../results/trained_models/{args.num_edges}_{args.zero_edges}_best_model.pth'))
        nbody_data = NBody(num_samples=args.num_samples, batch_size=args.batch_size)
        test_loader = nbody_data.test_loader()
        criterion = nn.MSELoss()
        test_loss = test_model(model, test_loader, criterion)
        print(f'Test Loss: {test_loss}')
        return

    metric = [1, 1, 1]
    clifford_algebra = CliffordAlgebra(metric)

    # Create the nbody_model
    model = NBodyTransformer(
        input_dim=3,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        clifford_algebra=clifford_algebra,
        num_edges=args.num_edges,
        zero_edges=args.zero_edges
    )
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    nbody_data = NBody(num_samples=args.num_samples, batch_size=args.batch_size)
    train_loader = nbody_data.train_loader()
    val_loader = nbody_data.val_loader()
    test_loader = nbody_data.test_loader()  # Assuming you have a test loader

    steps_per_epoch = len(train_loader)
    steps = args.epochs * steps_per_epoch

    scheduler = CosineAnnealingLR(optimizer, steps)

    best_val_loss = float('inf')
    early_stopping_counter = 0

    train_losses = []
    val_losses = []

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scheduler)
        val_loss = validate_epoch(model, val_loader, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Save nbody_model if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'./{args.num_edges}_{args.zero_edges}_best_model.pth')
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        # Stop training if validation loss has not improved for early_stopping_limit epochs
        if early_stopping_counter >= args.early_stopping_limit:
            print('Early stopping...')
            break

        print(f'Epoch {epoch + 1}, Training Loss: {train_loss}, Validation Loss: {val_loss}')

    # Load the best nbody_model and test it
    model.load_state_dict(torch.load(f'./{args.num_edges}_{args.zero_edges}_best_model.pth'))
    test_loss = test_model(model, test_loader, criterion)
    print(f'Test Loss: {test_loss}')
    # Save the training and validation losses to a CSV file
    save_losses_to_csv(args, train_losses, val_losses, test_loss)


if __name__ == '__main__':
    main()
