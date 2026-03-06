import os
import numpy as np
import torch
import random
import matplotlib.pyplot as plt

def check_directories(args):
    task_path = os.path.join(args.output_dir)
    if not os.path.exists(task_path):
        os.mkdir(task_path)
        print(f"Created {task_path} directory")
    
    folder = args.task
    
    save_path = os.path.join(task_path, folder)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        print(f"Created {save_path} directory")
    args.save_dir = save_path

    cache_path = os.path.join(args.input_dir, 'cache')
    if not os.path.exists(cache_path):
        os.mkdir(cache_path)
        print(f"Created {cache_path} directory")

    if args.debug:
        args.log_interval /= 10

    return args

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def setup_gpus(args):
    n_gpu = 0
    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
    args.n_gpu = n_gpu
    if n_gpu > 0:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    return args


def plot_losses(train_losses, val_losses, fname):
    """
    Plots the training and validation losses across epochs and saves the plot as an image file with name - fname(function argument). 

    Args:
        train_losses (list): List of training losses for each epoch.
        val_losses (list): List of validation losses for each epoch.
        fname (str): Name of the file to save the plot (without extension).

    Returns:
        None
    """

    # Create 'plots' directory if it doesn't exist
    if not os.path.isdir('plots'):
        os.mkdir('plots')

    # Plotting training and validation losses
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss per Epoch')

    # Saving the plot as an image file in 'plots' directory
    plt.savefig("./plots/" + fname + ".png")



def plot_metric(train_values, val_values, metric_name, fname):
    """
    Plots training and validation metrics across epochs and saves the plot.

    Args:
        train_values (list): Training metric values per epoch.
        val_values (list): Validation metric values per epoch.
        metric_name (str): Name of the metric (e.g., 'Loss', 'Accuracy').
        fname (str): File name to save the plot (without extension).

    Returns:
        None
    """
    # Ensure 'plots' directory exists
    os.makedirs('plots', exist_ok=True)

    # Plot training and validation values
    plt.figure(figsize=(8, 6))
    plt.plot(train_values, label=f'Training {metric_name}')
    plt.plot(val_values, label=f'Validation {metric_name}')

    # Adding labels and title
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} per Epoch')
    plt.legend()

    # Save the plot
    plt.savefig(f"./plots/{fname}.png")
    plt.close()