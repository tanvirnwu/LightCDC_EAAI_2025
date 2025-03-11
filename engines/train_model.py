import torch.nn as nn
from timeit import default_timer as timer
from torch.optim.lr_scheduler import StepLR
from utils.dataloader import *
from config import configs
import engines
import utils
def train_model(model_object: torch.nn.Module, model_name: str, batch_size: int, lr=None):
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    learning_rate = lr
    train_loader, val_loader = data_preparation_function(batch_size = batch_size)
    test_loader, class_to_idx = test_loader_function(batch_size = batch_size)

    print('\n=================================================================================')
    print(
        f'Train Dataset        |   Total Batches: {len(train_loader)}   |   Total Number of Images: {len(train_loader.dataset)}')
    print(
        f'Validation Dataset   |   Total Batches: {len(val_loader)}    |   Total Number of Images: {len(val_loader.dataset)}')
    print(
        f'Test Dataset         |   Total Batches: {len(test_loader)}    |   Total Number of Images: {len(test_loader.dataset)}')
    print('=================================================================================')
    print(
        f'Total                |   Total Batches: {len(train_loader) + len(val_loader) + len(test_loader)}   |   Total number of Images: {len(train_loader.dataset) + len(val_loader.dataset) + len(test_loader.dataset)}')
    print('=================================================================================')
    print(f'The dataset has total {len(class_to_idx)} classes | Names of the Classes: {list(class_to_idx.keys())}')
    print('=================================================================================')
    print(f'Now the model is training on {device} device      | Training Model: {model_name}')
    print('=================================================================================')

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model_object.parameters(), lr=learning_rate)


    # L2 Norm Regularization
    optimizer = torch.optim.Adam(params = model_object.parameters(), lr=learning_rate, weight_decay=l2_lambda)

    # LEARNING RATE DECAY
    if configs.scheduler_activate:
        scheduler = StepLR(optimizer, step_size=lr_decay_steps, gamma=gamma)
    else:
        scheduler = None

    # Start the timer
    start_time = timer()
    model_weights_saving_path = r'F:\Research\CDC5k\Storage\Saved_Models\\' + model_name + '.pth'

    model_results = engines.train(model=model_object,
                          train_dataloader=train_loader,
                          val_dataloader=val_loader, optimizer=optimizer, scheduler = scheduler,
                          gamma_value_increase_rate=gamma_value_increase_rate,
                          loss_fn=loss_fn, epochs=num_epochs,
                          model_path=model_weights_saving_path)
    end_time = timer()
    print(f"Total training time: {end_time - start_time:.3f} seconds")

    model_trianing_results_saving_path = r'F:\Research\CDC5k\Storage\Loss_Curve_Results\\' + model_name + '.pkl'
    utils.plot_loss_curves(model_results, model_trianing_results_saving_path, Loss_Curves, model_name)
