#!/usr/bin/env python3
"""
Model Parallelism Implementation in PyTorch

This module demonstrates how to split a neural network across multiple GPUs
using model parallelism techniques to utilize multiple GPU resources efficiently.
"""

import time
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt


@dataclass
class TrainingStats:
    """
    Container for tracking model training statistics.

    Attributes:
        epoch_times: Time taken for each epoch in seconds
        losses: Loss value after each epoch
        device_memory: GPU memory usage stats by device
    """
    epoch_times: List[float]
    losses: List[float]
    device_memory: Dict[str, List[float]]


class ModelParallelNetwork(nn.Module):
    """
    A neural network that splits computation across multiple GPUs.

    This model distributes its layers across separate CUDA devices to demonstrate
    model parallelism. First part of the network runs on first GPU, while the 
    second part runs on second GPU.
    """

    def __init__(
            self,
            input_size: int = 1000,
            hidden_sizes: List[int] = [800, 500, 400, 200],
            output_size: int = 10,
            first_device: str = "cuda:0",
            second_device: str = "cuda:1"
    ) -> None:
        """
        Initialize the model parallel network.

        Args:
            input_size: Dimension of input features
            hidden_sizes: List of hidden layer dimensions
            output_size: Dimension of output (e.g., number of classes)
            first_device: Device for first part of the model (e.g., "cuda:0")
            second_device: Device for second part of the model (e.g., "cuda:1")
        """
        super(ModelParallelNetwork, self).__init__()

        # Validate device availability
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available. This model requires GPU support.")

        device_count = torch.cuda.device_count()
        if device_count < 2:
            raise RuntimeError(f"This model requires at least 2 GPUs, but found {device_count}")

        self.first_device = first_device
        self.second_device = second_device

        # Split point - middle of hidden layers
        split_idx = len(hidden_sizes) // 2

        # First part of the network on first GPU
        layers1 = []
        current_size = input_size

        for i in range(split_idx):
            layers1.append(nn.Linear(current_size, hidden_sizes[i]))
            layers1.append(nn.ReLU())
            current_size = hidden_sizes[i]

        self.first_part = nn.Sequential(*layers1).to(first_device)

        # Second part of the network on second GPU  
        layers2 = []
        for i in range(split_idx, len(hidden_sizes)):
            layers2.append(nn.Linear(current_size, hidden_sizes[i]))
            layers2.append(nn.ReLU())
            current_size = hidden_sizes[i]

        # Output layer
        layers2.append(nn.Linear(current_size, output_size))
        self.second_part = nn.Sequential(*layers2).to(second_device)

        # For tracking performance
        self.transfer_times: List[float] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model, transferring tensors between devices.

        Args:
            x: Input tensor on CPU or first device

        Returns:
            Model output tensor on second device
        """
        # Ensure input is on first device
        if x.device != torch.device(self.first_device):
            x = x.to(self.first_device)

        # First part of computation on first GPU
        first_output = self.first_part(x)

        # Measure device transfer time
        transfer_start = time.time()

        # Transfer intermediate result to second GPU
        second_input = first_output.to(self.second_device)

        transfer_time = time.time() - transfer_start
        self.transfer_times.append(transfer_time)

        # Second part of computation on second GPU
        output = self.second_part(second_input)

        return output

    def get_avg_transfer_time(self) -> float:
        """
        Calculate the average tensor transfer time between devices.

        Returns:
            Average transfer time in seconds
        """
        if not self.transfer_times:
            return 0.0
        return sum(self.transfer_times) / len(self.transfer_times)


class SyntheticDataset(Dataset):
    """
    Generate synthetic data for training/testing the model parallel network.

    This dataset creates random feature vectors and class labels to simulate
    real data without requiring disk I/O.
    """

    def __init__(self, num_samples: int = 10000, input_size: int = 1000, num_classes: int = 10) -> None:
        """
        Initialize the synthetic dataset with random data.

        Args:
            num_samples: Number of samples to generate
            input_size: Dimension of input features
            num_classes: Number of output classes
        """
        # Generate random feature vectors
        self.data = torch.randn(num_samples, input_size)

        # Generate random class labels
        self.targets = torch.randint(0, num_classes, (num_samples,))

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.

        Args:
            idx: Index of the sample to fetch

        Returns:
            Tuple of (feature_vector, class_label)
        """
        return self.data[idx], self.targets[idx]


def get_gpu_memory_usage(device_ids: List[str] = None) -> Dict[str, float]:
    """
    Get current GPU memory usage for specified devices.

    Args:
        device_ids: List of CUDA device identifiers (e.g., ["cuda:0", "cuda:1"])
                   If None, all available devices are checked

    Returns:
        Dictionary mapping device identifiers to memory usage in MB
    """
    if device_ids is None:
        device_count = torch.cuda.device_count()
        device_ids = [f"cuda:{i}" for i in range(device_count)]

    memory_usage = {}

    for device in device_ids:
        # Extract device index from identifier
        device_idx = int(device.split(':')[1])

        # Get allocated memory in bytes and convert to MB
        allocated = torch.cuda.memory_allocated(device_idx) / (1024 * 1024)
        memory_usage[device] = allocated

    return memory_usage


def train_model(
        model: nn.Module,
        train_loader: DataLoader,
        num_epochs: int = 5,
        learning_rate: float = 0.001,
        device_ids: List[str] = ["cuda:0", "cuda:1"]
) -> TrainingStats:
    """
    Train the model parallel network and measure performance.

    Args:
        model: Model parallel network to train
        train_loader: DataLoader for training data
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        device_ids: List of devices to monitor for memory usage

    Returns:
        TrainingStats object containing performance metrics
    """
    criterion = nn.CrossEntropyLoss().to(device_ids[1])  # Loss on second device where output is
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize tracking metrics
    stats = TrainingStats(
        epoch_times=[],
        losses=[],
        device_memory={device: [] for device in device_ids}
    )

    for epoch in range(num_epochs):
        epoch_start = time.time()
        running_loss = 0.0

        for inputs, targets in train_loader:
            optimizer.zero_grad()

            # Forward pass (automatically handles device transfers)
            outputs = model(inputs)

            # Ensure targets are on same device as outputs
            targets = targets.to(outputs.device)

            # Compute loss and backpropagate
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Record metrics for this epoch
        epoch_time = time.time() - epoch_start
        avg_loss = running_loss / len(train_loader)
        memory_usage = get_gpu_memory_usage(device_ids)

        # Update statistics
        stats.epoch_times.append(epoch_time)
        stats.losses.append(avg_loss)
        for device, usage in memory_usage.items():
            stats.device_memory[device].append(usage)

        # Print progress
        print(f"Epoch {epoch + 1}/{num_epochs}, "
              f"Loss: {avg_loss:.4f}, "
              f"Time: {epoch_time:.2f}s, "
              f"Avg Transfer Time: {model.get_avg_transfer_time() * 1000:.2f}ms")

        # Reset transfer times for next epoch
        model.transfer_times = []

    return stats


def plot_training_stats(stats: TrainingStats, save_path: Optional[str] = None) -> None:
    """
    Visualize training statistics with plots.

    Args:
        stats: TrainingStats object containing performance metrics
        save_path: Path to save the plot image (if None, plot is displayed)
    """
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # Plot training loss
    axs[0, 0].plot(stats.losses, 'b-o')
    axs[0, 0].set_title('Training Loss')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].grid(True)

    # Plot epoch times
    axs[0, 1].plot(stats.epoch_times, 'r-o')
    axs[0, 1].set_title('Epoch Training Time')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Time (s)')
    axs[0, 1].grid(True)

    # Plot GPU memory usage
    for device, memory_usage in stats.device_memory.items():
        axs[1, 0].plot(memory_usage, marker='o', label=device)

    axs[1, 0].set_title('GPU Memory Usage')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('Memory (MB)')
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # Empty plot for comparison or future use
    axs[1, 1].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def compare_with_single_gpu(
        input_size: int = 1000,
        hidden_sizes: List[int] = [800, 500, 400, 200],
        output_size: int = 10,
        batch_size: int = 64,
        num_samples: int = 5000,
        num_epochs: int = 3
) -> Dict[str, Any]:
    """
    Compare model parallelism with single GPU approach.

    Args:
        input_size: Input dimension size
        hidden_sizes: List of hidden layer sizes
        output_size: Output dimension size
        batch_size: Mini-batch size for training
        num_samples: Number of synthetic samples
        num_epochs: Number of training epochs

    Returns:
        Dictionary with performance comparison metrics
    """
    # Create synthetic dataset
    dataset = SyntheticDataset(num_samples=num_samples, input_size=input_size, num_classes=output_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    results = {}

    # Approach 1: Model parallelism across two GPUs
    print("\n=== Training with Model Parallelism (2 GPUs) ===")
    model_parallel = ModelParallelNetwork(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        output_size=output_size
    )

    mp_stats = train_model(model_parallel, dataloader, num_epochs=num_epochs)
    results["model_parallel"] = {
        "avg_epoch_time": sum(mp_stats.epoch_times) / len(mp_stats.epoch_times),
        "final_loss": mp_stats.losses[-1],
        "stats": mp_stats
    }

    # Approach 2: Single GPU (all on cuda:0)
    print("\n=== Training on Single GPU (CUDA:0) ===")

    # Create a similar model, but all on first GPU
    layers = []
    current_size = input_size

    for hidden_size in hidden_sizes:
        layers.append(nn.Linear(current_size, hidden_size))
        layers.append(nn.ReLU())
        current_size = hidden_size

    layers.append(nn.Linear(current_size, output_size))
    single_gpu_model = nn.Sequential(*layers).to("cuda:0")

    # Custom training loop for single GPU
    criterion = nn.CrossEntropyLoss().to("cuda:0")
    optimizer = optim.Adam(single_gpu_model.parameters(), lr=0.001)

    # Initialize tracking metrics
    stats = TrainingStats(
        epoch_times=[],
        losses=[],
        device_memory={"cuda:0": [], "cuda:1": []}
    )

    for epoch in range(num_epochs):
        epoch_start = time.time()
        running_loss = 0.0

        for inputs, targets in dataloader:
            inputs = inputs.to("cuda:0")
            targets = targets.to("cuda:0")

            optimizer.zero_grad()
            outputs = single_gpu_model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Record metrics for this epoch
        epoch_time = time.time() - epoch_start
        avg_loss = running_loss / len(dataloader)
        memory_usage = get_gpu_memory_usage(["cuda:0", "cuda:1"])

        # Update statistics
        stats.epoch_times.append(epoch_time)
        stats.losses.append(avg_loss)
        for device, usage in memory_usage.items():
            stats.device_memory[device].append(usage)

        # Print progress
        print(f"Epoch {epoch + 1}/{num_epochs}, "
              f"Loss: {avg_loss:.4f}, "
              f"Time: {epoch_time:.2f}s")

    results["single_gpu"] = {
        "avg_epoch_time": sum(stats.epoch_times) / len(stats.epoch_times),
        "final_loss": stats.losses[-1],
        "stats": stats
    }

    # Calculate speedup/efficiency
    speedup = results["single_gpu"]["avg_epoch_time"] / results["model_parallel"]["avg_epoch_time"]
    results["speedup"] = speedup

    print("\n=== Performance Comparison ===")
    print(f"Model Parallel Avg Epoch Time: {results['model_parallel']['avg_epoch_time']:.2f}s")
    print(f"Single GPU Avg Epoch Time: {results['single_gpu']['avg_epoch_time']:.2f}s")
    print(f"Speedup: {speedup:.2f}x")

    return results


def main() -> None:
    """Main execution function to demonstrate model parallelism."""
    print("🚀 PyTorch Model Parallelism Benchmark")
    print("======================================")

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available. This script requires GPU support.")
        return

    gpu_count = torch.cuda.device_count()
    if gpu_count < 2:
        print(f"❌ This script requires at least 2 GPUs, but found {gpu_count}")
        return

    print(f"✓ Found {gpu_count} GPU devices:")
    for i in range(gpu_count):
        print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")

    # Configuration
    input_size = 1000
    hidden_sizes = [800, 500, 400, 200]
    output_size = 10
    batch_size = 64
    num_samples = 5000
    num_epochs = 3

    # Run performance comparison
    results = compare_with_single_gpu(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        output_size=output_size,
        batch_size=batch_size,
        num_samples=num_samples,
        num_epochs=num_epochs
    )

    # Plot results
    plot_training_stats(results["model_parallel"]["stats"], save_path="model_parallel_stats.png")
    plot_training_stats(results["single_gpu"]["stats"], save_path="single_gpu_stats.png")

    print("\n✅ Benchmark complete!")
    print("📊 Training statistics have been saved as PNG files.")


if __name__ == "__main__":
    main()
