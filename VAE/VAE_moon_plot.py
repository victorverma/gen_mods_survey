import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import seaborn as sns
from sklearn.datasets import make_moons
from sklearn.neighbors import KernelDensity

def plot_contours(data, save_plots):
    n_samples = 10000
    W, _ = make_moons(n_samples=n_samples, noise=0.1)
    kde = KernelDensity(bandwidth=0.2, kernel='gaussian')
    kde.fit(W)
    
    w_min, w_max = W[:, 0].min() - 0.5, W[:, 0].max() + 0.5
    z_min, z_max = W[:, 1].min() - 0.5, W[:, 1].max() + 0.5
    ww, zz = np.meshgrid(np.linspace(w_min, w_max, 100), np.linspace(z_min, z_max, 100))
    grid_samples = np.vstack([ww.ravel(), zz.ravel()]).T
    
    log_density = kde.score_samples(grid_samples)
    density = np.exp(log_density).reshape(ww.shape)
    plt.figure(figsize=(10, 8))
    plt.contourf(ww, zz, density, levels=20, cmap="Blues", alpha=0.5)
    
    plt.scatter(data[1:1000, 0], data[1:1000, 1], color="blue", marker="o", edgecolor="k", alpha=0.3, s=20, label="Synthetic data")
    plt.colorbar(label="Density")
    plt.title("Density Plot with Observed Data Scatter Overlay")
    plt.xlabel("x1")
    plt.ylabel("x2")
    if save_plots:
        plt.savefig("observed_data_plot.jpg", format="jpg", dpi=300)
        plt.close()
    else:
        plt.show()
    

def plot_temp(generated_data, save_plots):
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    kde = sns.kdeplot(x=generated_data[:, 0], y=generated_data[:, 1], cmap="inferno",
                      fill=True, thresh=0, levels=100, ax=ax)
    norm = Normalize(vmin=0, vmax=1)
    sm = ScalarMappable(cmap="inferno", norm=norm)
    sm.set_array([])
    
    plt.colorbar(sm, label="Density", ax=ax)
    plt.title("Temperature Plot of Simulated Half-Moon Data")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    
    if save_plots:
        plt.savefig("generated_temp_plot.jpg", format="jpg", dpi=300)
        plt.close()
    else:
        plt.show()
    
def plot_generated_data(data, generated_data, save_plots):
    kde = KernelDensity(bandwidth=0.2, kernel='gaussian')
    kde.fit(data)

    x_min, x_max = min(data[:, 0].min(), generated_data[:, 0].min()) - 0.5, max(data[:, 0].max(), generated_data[:, 0].max()) + 0.5
    y_min, y_max = min(data[:, 1].min(), generated_data[:, 1].min()) - 0.5, max(data[:, 1].max(), generated_data[:, 1].max()) + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid_samples = np.vstack([xx.ravel(), yy.ravel()]).T
    log_density = kde.score_samples(grid_samples)
    density = np.exp(log_density).reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, density, levels=20, cmap="Blues", alpha=0.5)
    plt.scatter(generated_data[:, 0], generated_data[:, 1], color='red', marker='x', alpha=0.5, label="Generated Data")

    plt.legend()
    plt.title("Density Plot with Generated Data Scatter Overlay")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.colorbar(label="Density")
    if save_plots:
        plt.savefig("generated_scatter_plot.jpg", format="jpg", dpi=300)
        plt.close()
    else:
        plt.show()
    

def get_latent_representation(model, data, device="cpu"):
    model.to(device)
    data = data.to(device)
    model.eval()
    with torch.no_grad():
        mu, logvar = model.encode(data)
        mu = mu.cpu().numpy()
        logvar = logvar.cpu().numpy()
    return mu, logvar


def plot_latent_parameters(mu, logvar, save_plots, labels=None):
    plt.figure(figsize=(10, 5))
    plt.suptitle("Latent Parameters Representation - $\mu$ and $\log\sigma^2$", fontsize=16)
    
    plt.subplot(1, 2, 1)
    scatter_mu = plt.scatter(mu[:, 0], mu[:, 1], c=labels, cmap='viridis', marker='o', alpha=0.3)
    plt.colorbar(scatter_mu, label="Cluster Label")
    plt.title("Plot for mu")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    
    plt.subplot(1, 2, 2)
    scatter_logvar = plt.scatter(logvar[:, 0], logvar[:, 1], c=labels, cmap='viridis', marker='o', alpha=0.3)
    plt.colorbar(scatter_logvar, label="Cluster Label")
    plt.title("Plot for logvar")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    
    plt.tight_layout()
    if save_plots:
        plt.savefig("latent_parameters_plot.jpg", format="jpg", dpi=300)
        plt.close()
    else:
        plt.show()
    
def plot_latent_variable(mu, logvar, save_plots, labels=None):
    std = np.exp(0.5 * logvar)
    z = mu + std * np.random.randn(*mu.shape)

    plt.figure(figsize=(10, 6))
    scatter_z = plt.scatter(z[:, 0], z[:, 1], c=labels, cmap='viridis', marker='o', alpha=0.3)
    plt.colorbar(scatter_z, label="Cluster Label")
    plt.title("Scatterplot of Latent Variable $z$")
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")

    plt.tight_layout()
    if save_plots:
        plt.savefig("latent_variable_plot.jpg", format="jpg", dpi=300)
        plt.close()
    else:
        plt.show()
