
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

def main():
    # Step 1: Get the current working directory
    current_directory = os.getcwd()
    print(f"Current directory: {current_directory}")
    
    # Step 2: Ensure data directory is accessible
    data_dir = os.path.join(current_directory, "data")
    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        return
    
    # Step 3: Load the data
    try:
        X = np.load(os.path.join(data_dir, 'X.npy'))
        y = np.load(os.path.join(data_dir, 'z.npy'))
        print(f"Loaded data: X shape = {X.shape}, y shape = {y.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Step 4: Convert to PyTorch tensor
    data = torch.tensor(X, dtype=torch.float32)
    print(f"Observed data size: {data.shape[0]}")
    
    # Step 5: Save a sample plot
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis", alpha=0.7)
    plt.colorbar(label="Class")
    plt.title("Sample Data Visualization")
    plot_path = os.path.join(current_directory, "sample_plot.png")
    plt.savefig(plot_path)
    print(f"Plot saved to: {plot_path}")
    plt.show()

if __name__ == "__main__":
    main()
