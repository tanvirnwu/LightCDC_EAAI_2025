import matplotlib.pyplot as plt
from typing import Dict, List
import pickle
import numpy as np
import os

def plot_loss_curves(results: Dict[str, List[float]], model_trianing_results_saving_path: str,
                     Loss_Curves: str, model_name):
    """Plots training curves of a results dictionary and saves the plots and results."""

    # Figure out how many epochs there were
    epochs = range(len(results['train_loss']))

    # Setup a plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

    # Plot loss
    ax1.plot(epochs, results['train_loss'], label='train_loss')
    ax1.plot(epochs, results['val_loss'], label='val_loss')
    ax1.set_title('Loss')
    ax1.set_xlabel('Epochs')
    ax1.legend()

    # Plot accuracy
    ax2.plot(epochs, results['train_acc'], label='train_accuracy')
    ax2.plot(epochs, results['val_acc'], label='val_accuracy')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.legend()

    # Save the figure to the provided path
    # fig.savefig(Loss_Curves)
    fig.savefig(os.path.join(Loss_Curves, f'{model_name}_Loss_Curve.jpg'))
    plt.close(fig)  # Close the plot to free memory

    # Save the results as a pickle file
    with open(model_trianing_results_saving_path, 'wb') as file:
        pickle.dump(results, file)

    print(f"Model results and plots saved to {model_trianing_results_saving_path} and {Loss_Curves}")



def display_image(image_tensor):
    """
    Displays a tensor as an image.

    Parameters:
    image_tensor (torch.Tensor): The image tensor to display.
    """
    # Check if the image tensor is on GPU, and move it to CPU if necessary
    if image_tensor.is_cuda:
        image_tensor = image_tensor.cpu()

    # Convert the tensor to a NumPy array
    image_numpy = image_tensor.numpy()

    # If the image has a channel dimension in the first position, move it to the last position
    if image_numpy.shape[0] == 3:
        image_numpy = np.transpose(image_numpy, (1, 2, 0))

    # Normalize the pixel values to [0, 1] for display
    image_numpy = image_numpy - image_numpy.min()
    image_numpy = image_numpy / image_numpy.max()

    # Display the image
    plt.imshow(image_numpy)
    plt.axis('off')  # Hide the axis ticks and labels
    # plt.show()
