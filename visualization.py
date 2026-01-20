import matplotlib.pyplot as plt
import numpy as np
import os

def _ensure_dir(path):
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)

def plot_training_metrics(mse_history, classification_error_history, title="Training Progress"):
    """
    Plots MSE and classification error during training.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # MSE plot
    ax1.plot(mse_history, label='MSE', color='blue', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Mean Squared Error')
    ax1.set_title(f'{title} - MSE')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Classification error plot
    ax2.plot(classification_error_history, label='Classification Error', color='red', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Classification Error Rate')
    ax2.set_title(f'{title} - Classification Error')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f'{title.replace(" ", "_").lower()}_metrics.png', dpi=300)
    plt.show()

def plot_weights_evolution(weights_history, layer_sizes, save_path=None):
    """
    Plots the evolution of weights during training for each layer.
    """
    weights_array = np.array(weights_history)
    num_weights = weights_array.shape[1]
    
    fig, axes = plt.subplots(len(layer_sizes) - 1, 1, figsize=(12, 4 * (len(layer_sizes) - 1)))
    
    if len(layer_sizes) == 2:
        axes = [axes]
    
    weight_idx = 0
    for layer_idx in range(len(layer_sizes) - 1):
        num_layer_weights = layer_sizes[layer_idx] * layer_sizes[layer_idx + 1]
        
        for i in range(num_layer_weights):
            axes[layer_idx].plot(weights_array[:, weight_idx + i], 
                                label=f'w{i+1}', alpha=0.7, linewidth=1.5)
            
        axes[layer_idx].set_xlabel('Epoch')
        axes[layer_idx].set_ylabel('Weight Value')
        axes[layer_idx].set_title(f'Layer {layer_idx + 1} Weights Evolution ({layer_sizes[layer_idx]}→{layer_sizes[layer_idx + 1]})')
        axes[layer_idx].grid(True, alpha=0.3)
        axes[layer_idx].legend(loc='best', ncol=3, fontsize=8)
        
        weight_idx += num_layer_weights
    
    plt.tight_layout()
    _ensure_dir(save_path)
    plt.savefig(save_path or 'weights_evolution.png', dpi=300)
    plt.show()

def plot_weights_evolution_separate(weights_history, layer_sizes, save_dir=None, legend_limit=20):
    """
    Plots weight evolution in separate figures per layer.
    Only the first `legend_limit` weights are labeled to keep the legend readable.
    """
    weights_array = np.array(weights_history)
    weight_idx = 0

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    for layer_idx in range(len(layer_sizes) - 1):
        num_layer_weights = layer_sizes[layer_idx] * layer_sizes[layer_idx + 1]
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))

        for i in range(num_layer_weights):
            label = f"w{i+1}" if i < legend_limit else None
            ax.plot(
                weights_array[:, weight_idx + i],
                label=label,
                alpha=0.7,
                linewidth=1.2
            )

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Weight Value')
        ax.set_title(f'Layer {layer_idx + 1} Weights Evolution ({layer_sizes[layer_idx]}→{layer_sizes[layer_idx + 1]})')
        ax.grid(True, alpha=0.3)
        if legend_limit > 0:
            ax.legend(loc='best', ncol=3, fontsize=8)

        plt.tight_layout()
        filename = f'weights_layer{layer_idx + 1}.png'
        save_path = os.path.join(save_dir, filename) if save_dir else filename
        _ensure_dir(save_path)
        plt.savefig(save_path, dpi=300)
        plt.close(fig)

def plot_loss_and_accuracy(loss_history, accuracy_history, title="Model Training", save_path=None):
    """
    Plots loss and accuracy for neural network training (e.g., Titanic).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    ax1.plot(loss_history, label='Loss (BCE)', color='orange', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{title} - Loss')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Accuracy plot
    ax2.plot(accuracy_history, label='Accuracy', color='green', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title(f'{title} - Accuracy')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_ylim([0, 1.1])
    
    plt.tight_layout()
    _ensure_dir(save_path)
    plt.savefig(save_path or f'{title.replace(" ", "_").lower()}_training.png', dpi=300)
    plt.show()

def plot_loss_accuracy_and_classification_error(loss_history, accuracy_history, classification_error_history, title="Model Training", save_path=None):
    """
    Plots loss, accuracy, and classification error for neural network training.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(loss_history, label='Loss (BCE)', color='orange', linewidth=2)
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].set_title(f'{title} - Loss'); axes[0].grid(True, alpha=0.3); axes[0].legend()

    axes[1].plot(accuracy_history, label='Accuracy', color='green', linewidth=2)
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy')
    axes[1].set_title(f'{title} - Accuracy'); axes[1].grid(True, alpha=0.3); axes[1].legend(); axes[1].set_ylim([0, 1.1])

    axes[2].plot(classification_error_history, label='Classification Error', color='red', linewidth=2)
    axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('Error Rate')
    axes[2].set_title(f'{title} - Classification Error'); axes[2].grid(True, alpha=0.3); axes[2].legend(); axes[2].set_ylim([0, 1.1])

    plt.tight_layout()
    _ensure_dir(save_path)
    plt.savefig(save_path or f'{title.replace(" ", "_").lower()}_loss_acc_class_error.png', dpi=300)
    plt.show()

def plot_xor_decision_boundary(X, Y, W, B, forward_func):
    """
    Plots XOR decision boundary with data points.
    """
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    
    Z = np.zeros_like(xx)
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            point = np.array([xx[i, j], yy[i, j]])
            y_hat, _ = forward_func(point, W, B, [None] * len(W))
            Z[i, j] = y_hat.item()
    
    plt.figure(figsize=(8, 8))
    plt.contourf(xx, yy, Z, levels=20, cmap='RdYlBu', alpha=0.8)
    plt.colorbar(label='Output')
    
    colors = ['blue' if y == 0 else 'red' for y in Y]
    plt.scatter(X[:, 0], X[:, 1], c=colors, s=200, edgecolors='black', linewidth=2, zorder=5)
    
    plt.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=3)
    
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('XOR Problem - Decision Boundary')
    plt.grid(True, alpha=0.3)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    
    plt.savefig('xor_decision_boundary.png', dpi=300)
    plt.show()