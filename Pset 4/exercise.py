import math
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import wandb

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

def get_dataloader(is_train, batch_size, slice = 5):
    full_dataset = torchvision.datasets.MNIST(root = ".", train = is_train, transform = T.ToTensor(), 
                                              download = True)
    sub_dataset = torch.utils.data.Subset(full_dataset, indices = range(0, len(full_dataset), slice))
    loader = torch.utils.data.DataLoader(dataset = sub_dataset, batch_size = batch_size,
                                         shuffle = is_train, pin_memory = True, num_workers = 2)
    return loader

def get_model(dropout):
    "A simple model"
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(256, 10)).to(DEVICE)
    return model

def validate_model(model, valid_dl, loss_func, dropout, log_images = False, batch_idx = 0):
    model.eval()
    val_loss = 0.
    # Exercise 2: set up table
    # assumes that wandb has been logged into
    wandb.init()
    wandb_table = wandb.Table(columns = ['images', 'labels', 'predictions'])
    with torch.inference_mode():
        correct = 0
        for i, (images, labels) in enumerate(valid_dl):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # Forward pass
            outputs = model(images)
            val_loss += loss_func(outputs, labels).item() * labels.size(0)

            # Compute accuracy and accumulate
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

            # Exercise 2: add rows to table
            for i in range(images.shape[0]):
                wandb_table.add_data(wandb.Image(images[i].numpy()), labels[i].item(),
                                     predicted[i].item())
    # Exercise 2: save the table if log_images argument is true
    if log_images:
        wandb.log({f"validation_table_dropout_{dropout}": wandb_table})
    return val_loss / len(valid_dl.dataset), correct / len(valid_dl.dataset)

class Config():
    """Helper to convert a dictionary to a class"""
    def __init__(self, dict):
        "A simple config class"
        self.epochs = dict['epochs']
        self.batch_size = dict['batch_size']
        self.lr = dict['lr']
        self.dropout = dict['dropout']

def train():
    wandb.login()
    # Exercise 4: Configure sweep and setup helper function to call in sweep
    common_config_dict = {
        "epochs": 10,
        "batch_size": 128,
        "lr": 1e-3,
    }
    sweep_configuration = {
        "name": "dropout-sweep",
        "project": "pset 4 experiments",
        "metric": {"name": "accuracy", "goal": "maximize"},
        "method": "grid",
        "parameters": {
            **{k: {"value": v} for k, v in common_config_dict.items()},
            "dropout": {"values": [0.01, 0.2, 0.4, 0.6, 0.8]}
        }
    }
    models = []
    def train_one_model():    
        # Exercise 4: retrieve current dropout value
        wandb.init()
        dropout = wandb.config.dropout
        config_dict = {**common_config_dict, "dropout": dropout}
        config = Config(config_dict)

        # Get the data
        train_dl = get_dataloader(is_train = True, batch_size = config.batch_size)
        valid_dl = get_dataloader(is_train = False, batch_size = 2 * config.batch_size)
        n_steps_per_epoch = math.ceil(len(train_dl.dataset) / config.batch_size)
        
        # A simple MLP model
        model = get_model(config.dropout)

        # Make the loss and optimizer
        loss_func = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

        # Training
        example_ct = 0
        step_ct = 0
        for epoch in range(config.epochs):
            model.train()
            for step, (images, labels) in enumerate(train_dl):
                images, labels = images.to(DEVICE), labels.to(DEVICE)

                outputs = model(images)
                train_loss = loss_func(outputs, labels)
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
        
                example_ct += len(images)
                step_ct += 1

            val_loss, accuracy = validate_model(model, valid_dl, loss_func, config.dropout, 
                                                log_images=(epoch == (config.epochs-1)))

            # Exercise 1: log values with wandb instead of printing
            wandb.log({
                "Train Loss": float(f"{train_loss:.3f}"),
                "Valid Loss": float(f"{val_loss:3f}"),
                "Accuracy": float(f"{accuracy:.2f}")
            })
        # Exercise 3: save both model state dict and accuracy to determine best 3 later
        models.append((model.state_dict(), accuracy, config.dropout))

    # Exercise 4: run sweep with the same training function and different dropout vals
    sweep_id = wandb.sweep(sweep_configuration)
    wandb.agent(sweep_id, function = train_one_model)

    # Exercise 3: locally save best 3 models and then log wandb artifacts
    wandb.init(project = "pset 4 experiments")
    models.sort(key = lambda x: x[1])
    for i, (state_dict, acc, dropout) in enumerate(models[-3:]):
        filename = f"model_{3-i}_dropout_{dropout}_acc_{acc:.2f}.pt"
        model_path = f"best_models/{filename}"
        torch.save(state_dict, model_path)
        artifact = wandb.Artifact(filename, type = 'model', 
                                  metadata = {'Validation accuracy': acc, "Dropout": dropout})
        artifact.add_file(model_path)
        wandb.log_artifact(artifact)
    wandb.finish()

if __name__ == "__main__":
    train()