#!/usr/bin/env python3
"""
DataLoader Benchmarking Tool

This module provides utilities to benchmark PyTorch DataLoader performance
with different configurations to find optimal data loading parameters.
"""

import time
from typing import Dict, List, Tuple, Union, Optional
import multiprocessing
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt


class SimulatedDataset(Dataset):
    """
    Simulates data loading and preprocessing with configurable delay.

    Each item fetch is delayed by a set amount to emulate I/O and preprocessing
    overhead in real datasets.

    Attributes:
        size: Total number of samples in the dataset
        delay: Time in seconds to sleep for each item fetch (default: 0.01s)
    """

    def __init__(self, size: int = 10_000, delay: float = 0.01):
        """
        Initialize the simulated dataset.

        Args:
            size: Number of samples in the dataset
            delay: Time in seconds to sleep for each __getitem__ call
        """
        self.size = size
        self.delay = delay

    def __len__(self) -> int:
        """Return the total number of samples."""
        return self.size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Fetch a sample from the dataset with simulated delay.

        Args:
            idx: Index of the item to fetch

        Returns:
            A tuple containing a tensor of shape (3, 224, 224) and a class label
        """
        # Simulate loading overhead
        time.sleep(self.delay)

        # Generate synthetic data (3 channels, 224x224 image)
        data = torch.randn(3, 224, 224)
        label = np.random.randint(0, 10)

        return data, label


@dataclass
class BenchmarkResult:
    """
    Stores the results of a DataLoader benchmark run.

    Attributes:
        num_workers: Number of worker processes used
        batch_size: Batch size used for the test
        pin_memory: Whether pin_memory was enabled
        prefetch_factor: Prefetch factor used (if applicable)
        duration: Time taken to process the batches in seconds
        throughput: Items per second processed
    """
    num_workers: int
    batch_size: int
    pin_memory: bool
    prefetch_factor: Optional[int]
    duration: float
    throughput: float


def benchmark_dataloader(
        num_workers: int,
        batch_size: int = 64,
        num_batches: int = 100,
        pin_memory: bool = False,
        prefetch_factor: int = 2,
        device: Optional[str] = None
) -> BenchmarkResult:
    """
    Benchmark DataLoader performance with the given parameters.

    Args:
        num_workers: Number of worker processes to use
        batch_size: Number of samples per batch
        num_batches: Number of batches to process for timing
        pin_memory: Whether to use pinned memory
        prefetch_factor: Number of batches to prefetch per worker
        device: Target device for tensor transfer ('cuda', 'cpu', or None)

    Returns:
        BenchmarkResult containing performance metrics
    """
    dataset = SimulatedDataset()
    total_items = batch_size * num_batches

    # Configure DataLoader parameters
    loader_kwargs: Dict[str, Union[Dataset, int, bool]] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }

    # Add prefetch_factor only when using workers
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor

    # Create the DataLoader
    loader = DataLoader(**loader_kwargs)

    # Determine target device
    if device is None:
        device = "cuda" if torch.cuda.is_available() and pin_memory else "cpu"

    # Benchmark data loading
    start = time.perf_counter()

    for i, (data, labels) in enumerate(loader):
        if i >= num_batches:
            break

        # Simulate data transfer to target device
        if device == "cuda":
            data = data.to(device, non_blocking=True)
            labels = torch.tensor(labels).to(device, non_blocking=True)

    end = time.perf_counter()
    duration = end - start

    # Calculate throughput (items per second)
    throughput = total_items / duration if duration > 0 else 0

    return BenchmarkResult(
        num_workers=num_workers,
        batch_size=batch_size,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        duration=duration,
        throughput=throughput
    )


def run_benchmarks(
        max_workers: Optional[int] = None,
        pin_memory: bool = True,
        prefetch_factor: int = 2,
        batch_size: int = 64,
        num_batches: int = 100
) -> List[BenchmarkResult]:
    """
    Run a series of benchmarks with varying numbers of workers.

    Args:
        max_workers: Maximum number of workers to test (defaults to CPU count)
        pin_memory: Whether to use pinned memory
        prefetch_factor: Number of batches loaded in advance per worker
        batch_size: Number of samples per batch
        num_batches: Number of batches to process in each test

    Returns:
        List of BenchmarkResult objects for each worker count tested
    """
    if max_workers is None:
        max_workers = min(8, multiprocessing.cpu_count())

    results: List[BenchmarkResult] = []

    print(f"🔍 Running benchmarks with {num_batches} batches of size {batch_size}")
    print(f"📊 Configuration: pin_memory={pin_memory}, prefetch_factor={prefetch_factor}")
    print("-" * 60)

    # Test with different worker counts
    for num_workers in range(max_workers + 1):
        result = benchmark_dataloader(
            num_workers=num_workers,
            batch_size=batch_size,
            num_batches=num_batches,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor
        )

        print(f"Workers: {num_workers:2d} | "
              f"Duration: {result.duration:.2f}s | "
              f"Throughput: {result.throughput:.2f} items/s")

        results.append(result)

    return results


def plot_benchmark_results(results: List[BenchmarkResult], save_path: Optional[str] = None) -> None:
    """
    Visualize benchmark results with matplotlib.

    Args:
        results: List of BenchmarkResult objects
        save_path: Path to save the plot image (if None, plot is displayed)
    """
    workers = [r.num_workers for r in results]
    durations = [r.duration for r in results]
    throughputs = [r.throughput for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot duration vs workers
    ax1.plot(workers, durations, marker='o', linewidth=2, color='#1f77b4')
    ax1.set_title("Data Loading Time vs Worker Count")
    ax1.set_xlabel("Number of Workers")
    ax1.set_ylabel("Time (seconds)")
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Plot throughput vs workers
    ax2.plot(workers, throughputs, marker='s', linewidth=2, color='#2ca02c')
    ax2.set_title("Throughput vs Worker Count")
    ax2.set_xlabel("Number of Workers")
    ax2.set_ylabel("Throughput (items/second)")
    ax2.grid(True, linestyle='--', alpha=0.7)

    # Configuration details
    config_text = (
        f"Batch Size: {results[0].batch_size}\n"
        f"Pin Memory: {results[0].pin_memory}\n"
        f"Prefetch Factor: {results[0].prefetch_factor if results[0].prefetch_factor else 'N/A'}"
    )
    fig.text(0.5, 0.01, config_text, ha='center', fontsize=10)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def main() -> None:
    """Run the benchmark suite and visualize results."""
    print("🚀 DataLoader Performance Benchmark")
    print("=" * 60)

    # Run benchmarks
    results = run_benchmarks(
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=2,
        batch_size=64,
        num_batches=50
    )

    print("\n✅ Benchmark complete!")

    # Find optimal configuration
    best_result = min(results, key=lambda r: r.duration)

    print("\n🏆 Optimal Configuration:")
    print(f"   Workers: {best_result.num_workers}")
    print(f"   Duration: {best_result.duration:.2f} seconds")
    print(f"   Throughput: {best_result.throughput:.2f} items/second")

    # Plot results
    plot_benchmark_results(results)


if __name__ == "__main__":
    main()
