from config import configs
import torch.nn as nn
from tqdm.auto import tqdm
import torch


def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer):
    model.train()
    train_loss, train_acc = 0, 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(configs.device), y.to(configs.device)

        y_pred = model(X)

        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculating and accumulating accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)

    # Adjusting metrics to get average loss and accuracy per batch
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)

    return train_loss, train_acc


def val_step(model: torch.nn.Module,
             dataloader: torch.utils.data.DataLoader,
             loss_fn: torch.nn.Module):
    model.eval()
    val_loss, val_acc = 0, 0

    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(configs.device), y.to(configs.device)

            val_pred_logits = model(X)

            loss = loss_fn(val_pred_logits, y)
            val_loss += loss.item()

            val_pred_labels = val_pred_logits.argmax(dim=1)
            val_acc += ((val_pred_labels == y).sum().item() / len(val_pred_labels))

    val_loss = val_loss / len(dataloader)
    val_acc = val_acc / len(dataloader)
    return val_loss, val_acc


def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          val_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler._LRScheduler = None,
          gamma_value_increase_rate: float = None,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
          epochs: int = configs.num_epochs,
          model_path: str = "best_model_1.pth", patience: int = 10, min_delta: float = 0.0001):
    best_val_loss = float('inf')
    best_val_acc = 0.0
    early_stopping_counter = 0
    best_epoch = 0

    results = {"train_loss": [],
               "train_acc": [],
               "val_loss": [],
               "val_acc": []}

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer)
        val_loss, val_acc = val_step(model=model,
                                     dataloader=val_dataloader,
                                     loss_fn=loss_fn)

        # Early stopping and model checkpointing based on validation loss
        if val_loss <= best_val_loss - min_delta:
            best_val_loss = val_loss
            early_stopping_counter = 0

            # Save the model as it's an improvement
            torch.save(model.state_dict(), model_path)
            best_epoch = epoch + 1
            print(f"Model saved with val loss {val_loss:.4f} at epoch {best_epoch}.")

        else:
            # Increment early stopping counter if no improvement
            early_stopping_counter += 1
            if early_stopping_counter > patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

        # Update learning rate if scheduler is provided
        if scheduler:
            # Adjust gamma if conditions are met
            if gamma_value_increase_rate is not None and epoch % configs.lr_decay_steps == 0 and epoch > 0 and epoch <= configs.stop_gamma_value_increase_epoch:
                scheduler.gamma *= (1 + gamma_value_increase_rate)
                print(f"Gamma value increased to {scheduler.gamma} at epoch {epoch}.")
                print(f"Current Learning Rate: {scheduler.get_last_lr()}")

            scheduler.step()

        print(f"Epoch: {epoch + 1} | "
              f"train_loss: {train_loss:.4f} | "
              f"train_acc: {train_acc*100:.2f}% | "
              f"val_loss: {val_loss:.4f} | "
              f"val_acc: {val_acc*100:.2f}% |"
              f"ES Counter: {early_stopping_counter}")

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)

    print(f"The best model found at epoch {best_epoch} with Val Loss: {best_val_loss:.4f} | Val Accuracy: {best_val_acc*100:.2f}%.")

    return results














