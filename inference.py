import time
from typing import List, Tuple, Dict, Any, Optional, Union
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TextPromptDataset(Dataset):
    """
    A custom PyTorch dataset for processing text prompts for LLM inference.

    Attributes:
        prompts (List[str]): List of text prompts to process.
        tokenizer: HuggingFace tokenizer for encoding text.
        max_length (int): Maximum token length for each prompt.
    """

    def __init__(self, prompts: List[str], tokenizer, max_length: int = 50):
        """
        Initialize the dataset with text prompts.

        Args:
            prompts: List of strings containing the input prompts.
            tokenizer: HuggingFace tokenizer for the model.
            max_length: Maximum sequence length for tokenization.
        """
        self.prompts = prompts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        """Return the number of prompts in the dataset."""
        return len(self.prompts)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], str]:
        """
        Get a tokenized prompt at the specified index.

        Args:
            idx: Index of the prompt to retrieve.

        Returns:
            Tuple containing the tokenized inputs and original prompt.
        """
        prompt = self.prompts[idx]
        inputs = self.tokenizer(
            prompt,
            padding='max_length',
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt'
        )
        # Remove batch dimension to allow proper batching in DataLoader
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        return inputs, prompt


class SequentialInference:
    """
    Class for performing sequential LLM inference on multiple text prompts.

    This class handles loading a language model and processing prompts one by one
    or in small batches sequentially.

    Attributes:
        model_name (str): Name of the HuggingFace model to use.
        device (str): Device to run inference on ('cuda' or 'cpu').
        model: The loaded language model.
        tokenizer: The loaded tokenizer.
    """

    def __init__(self, model_name: str = "distilgpt2"):
        """
        Initialize the sequential inference system.

        Args:
            model_name: HuggingFace model identifier to load.
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

        # Load model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.tokenizer.padding_side = "left"

    def generate_response(self, prompt: str, max_length: int = 50) -> str:
        """
        Generate text from a single prompt.

        Args:
            prompt: Input text to generate from.
            max_length: Maximum token length for the generated output.

        Returns:
            Generated text string.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.eos_token_id
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def batch_inference(self, prompts: List[str], batch_size: int = 1) -> Tuple[List[str], float]:
        """
        Process multiple prompts sequentially in batches.

        Args:
            prompts: List of input text prompts.
            batch_size: Number of prompts to process in each batch.

        Returns:
            Tuple containing list of generated responses and total execution time.
        """
        results = []
        total_time = 0

        with tqdm(total=len(prompts), desc="Sequential processing") as pbar:
            for i in range(0, len(prompts), batch_size):
                batch = prompts[i:i + batch_size]
                batch_results = []

                # Time the batch processing
                start_time = time.time()

                for prompt in batch:
                    response = self.generate_response(prompt)
                    batch_results.append(response)

                end_time = time.time()
                batch_time = end_time - start_time
                total_time += batch_time

                results.extend(batch_results)
                pbar.update(len(batch))

        return results, total_time


class AccelerateInference:
    """
    Class for performing optimized parallel LLM inference using Hugging Face Accelerate.

    This class leverages the Accelerate library to efficiently utilize available
    hardware resources for faster inference across batches of prompts.

    Attributes:
        model_name (str): Name of the HuggingFace model to use.
        accelerator: Hugging Face Accelerate accelerator instance.
        model: The accelerated language model.
        tokenizer: The loaded tokenizer.
    """

    def __init__(self, model_name: str = "distilgpt2"):
        """
        Initialize the accelerated inference system.

        Args:
            model_name: HuggingFace model identifier to load.
        """
        self.model_name = model_name
        self.accelerator = Accelerator(mixed_precision='fp16')

        # Log execution environment details
        device_info = f"Using device: {self.accelerator.device}"
        if torch.cuda.is_available():
            device_info += f" ({torch.cuda.get_device_name(0)})"
            device_info += f", Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"

        logger.info(device_info)

        # Load model and tokenizer with optimizations
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.accelerator.device.type == "cuda" else None
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.padding_side = "left"

        # Set EOS token as pad token
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Prepare model for accelerated inference
        self.model = self.accelerator.prepare(self.model)

    def generate_responses(
            self,
            prompts: List[str],
            batch_size: int = 8,
            max_length: int = 50
    ) -> Tuple[List[str], float]:
        """
        Generate responses for multiple prompts using Accelerate-optimized batching.

        Args:
            prompts: List of input text prompts.
            batch_size: Number of prompts to process in each batch.
            max_length: Maximum token length for the generated outputs.

        Returns:
            Tuple containing list of generated responses and total execution time.
        """
        # Create dataset and dataloader
        dataset = TextPromptDataset(prompts, self.tokenizer, max_length)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=min(4, batch_size)  # Optimize dataloader workers
        )
        dataloader = self.accelerator.prepare(dataloader)

        results = []
        start_time = time.time()

        # Set model to evaluation mode
        self.model.eval()
        with torch.no_grad():
            for batch_inputs, original_prompts in tqdm(dataloader, desc="Accelerate processing"):
                # Generate text for the batch
                outputs = self.model.generate(
                    input_ids=batch_inputs['input_ids'],
                    attention_mask=batch_inputs['attention_mask'],
                    max_new_tokens=max_length,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=False  # Deterministic generation for benchmarking
                )

                # Decode the generated text
                batch_responses = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                results.extend(batch_responses)

        end_time = time.time()
        execution_time = end_time - start_time

        return results, execution_time


def run_experiment(
        num_prompts_list: List[int] = [10, 50, 100],
        batch_sizes: List[int] = [1, 4, 8, 16]
) -> Dict[str, Any]:
    """
    Run experiments comparing sequential and accelerated inference.

    Args:
        num_prompts_list: List of different prompt counts to test.
        batch_sizes: List of different batch sizes to test.

    Returns:
        Dictionary containing all experiment results.
    """
    results = {
        "sequential": {bs: [] for bs in batch_sizes},
        "accelerate": {bs: [] for bs in batch_sizes}
    }

    # Generate synthetic prompts
    max_prompts = max(num_prompts_list)
    prompts = [f"The future of artificial intelligence is {i}" for i in range(max_prompts)]

    # Sequential inference with different batch sizes
    logger.info("Initializing sequential inference model...")
    sequential_model = SequentialInference()

    for batch_size in batch_sizes:
        logger.info(f"Running sequential inference with batch_size={batch_size}")
        for num_prompts in num_prompts_list:
            current_prompts = prompts[:num_prompts]
            _, execution_time = sequential_model.batch_inference(current_prompts, batch_size)
            results["sequential"][batch_size].append((num_prompts, execution_time))
            logger.info(f"  Processed {num_prompts} prompts in {execution_time:.2f} seconds")

    # Accelerate inference with different batch sizes
    logger.info("Initializing accelerated inference model...")
    accelerate_model = AccelerateInference()

    for batch_size in batch_sizes:
        logger.info(f"Running accelerated inference with batch_size={batch_size}")
        for num_prompts in num_prompts_list:
            current_prompts = prompts[:num_prompts]
            _, execution_time = accelerate_model.generate_responses(current_prompts, batch_size)
            results["accelerate"][batch_size].append((num_prompts, execution_time))
            logger.info(f"  Processed {num_prompts} prompts in {execution_time:.2f} seconds")

    return results


def plot_results(results: Dict[str, Any], num_prompts_list: List[int]) -> None:
    """
    Create visualizations comparing sequential and accelerated inference performance.

    Args:
        results: Dictionary containing experiment result data.
    """
    # Plot 1: Sequential vs Accelerate (with fixed batch size)
    plt.figure(figsize=(12, 8))

    # Use the middle batch size for comparison
    batch_sizes = list(results["sequential"].keys())
    batch_size = batch_sizes[len(batch_sizes) // 2]

    # Plot sequential results
    seq_prompts, seq_times = zip(*results["sequential"][batch_size])
    plt.plot(seq_prompts, seq_times, marker='o', linewidth=2,
             label=f'Sequential (batch_size={batch_size})')

    # Plot accelerate results
    acc_prompts, acc_times = zip(*results["accelerate"][batch_size])
    plt.plot(acc_prompts, acc_times, marker='s', linewidth=2,
             label=f'Accelerate (batch_size={batch_size})')

    plt.xlabel('Number of Prompts', fontsize=12)
    plt.ylabel('Execution Time (seconds)', fontsize=12)
    plt.title('Sequential vs. Accelerate Inference Performance', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('sequential_vs_accelerate.png', dpi=300)

    # Plot 2: Effect of Batch Size on Accelerate Performance
    plt.figure(figsize=(12, 8))

    for batch_size in results["accelerate"]:
        prompts, times = zip(*results["accelerate"][batch_size])
        plt.plot(prompts, times, marker='D', linewidth=2, label=f'Batch Size = {batch_size}')

    plt.xlabel('Number of Prompts', fontsize=12)
    plt.ylabel('Execution Time (seconds)', fontsize=12)
    plt.title('Effect of Batch Size on Accelerate Performance', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('batch_size_effect_accelerate.png', dpi=300)

    # Plot 3: Speedup Analysis
    plt.figure(figsize=(12, 8))

    # Calculate and plot speedup for each batch size
    for batch_size in batch_sizes:
        speedups = []
        prompts_list = []

        for i, (num_prompts, seq_time) in enumerate(results["sequential"][batch_size]):
            acc_time = results["accelerate"][batch_size][i][1]
            speedup = seq_time / acc_time
            speedups.append(speedup)
            prompts_list.append(num_prompts)

        plt.plot(prompts_list, speedups, marker='x', linewidth=2, label=f'Batch Size = {batch_size}')

    plt.axhline(y=1, linestyle='--', alpha=0.5, color='gray')
    plt.text(max(prompts_list), 1, 'Sequential baseline',
             verticalalignment='bottom', horizontalalignment='right')

    plt.xlabel('Number of Prompts', fontsize=12)
    plt.ylabel('Speedup Factor (Sequential Time / Accelerate Time)', fontsize=12)
    plt.title('Accelerate Speedup Analysis', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('speedup_analysis_accelerate.png', dpi=300)

    # Plot 4: Comparison bar chart for the largest prompt set
    plt.figure(figsize=(10, 8))

    largest_prompt_set = max(num_prompts_list)
    seq_times = []
    acc_times = []

    for batch_size in batch_sizes:
        seq_idx = [i for i, (n, _) in enumerate(results["sequential"][batch_size]) if n == largest_prompt_set][0]
        acc_idx = [i for i, (n, _) in enumerate(results["accelerate"][batch_size]) if n == largest_prompt_set][0]

        seq_times.append(results["sequential"][batch_size][seq_idx][1])
        acc_times.append(results["accelerate"][batch_size][acc_idx][1])

    x = np.arange(len(batch_sizes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 8))
    rects1 = ax.bar(x - width / 2, seq_times, width, label='Sequential', alpha=0.8)
    rects2 = ax.bar(x + width / 2, acc_times, width, label='Accelerate', alpha=0.8)

    ax.set_ylabel('Execution Time (seconds)', fontsize=12)
    ax.set_xlabel('Batch Size', fontsize=12)
    ax.set_title(f'Sequential vs Accelerate Performance (n={largest_prompt_set})', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(batch_sizes)
    ax.legend(fontsize=12)

    # Add execution time labels on top of bars
    def add_labels(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}s',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

    add_labels(rects1)
    add_labels(rects2)

    fig.tight_layout()
    plt.savefig('batch_comparison.png', dpi=300)


def main() -> None:
    """Main entry point for the benchmark application."""
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Define experiment parameters
    num_prompts_list = [10, 30, 50]  # Increase based on available resources
    batch_sizes = [1, 4, 8, 16]  # Different batch sizes to test

    # Run experiments
    logger.info("Starting LLM inference experiments...")
    results = run_experiment(num_prompts_list, batch_sizes)

    # Plot and save results
    logger.info("Generating result visualizations...")
    plot_results(results, num_prompts_list)

    logger.info("Experiments completed! Check the generated plots.")


if __name__ == "__main__":
    main()
