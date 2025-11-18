from typing import Optional
import os
import json
import numpy as np
import random

from argdantic import ArgParser
from pydantic import BaseModel
from tqdm import tqdm
from sympy import randprime, isprime

try:
    from dataset.common import PuzzleDatasetMetadata
except ModuleNotFoundError:  # Backwards compat when running as a script
    from common import PuzzleDatasetMetadata  # type: ignore


cli = ArgParser()


class DataProcessConfig(BaseModel):
    output_dir: str = "data/factorization-semiprimes"
    
    train_size: int = 10000
    test_size: int = 1000
    
    min_bits: int = 16  # Minimum number of bits for the primes
    max_bits: int = 32  # Maximum number of bits for the primes
    
    max_seq_len: int = 64  # Maximum sequence length for binary representation
    
    seed: int = 42


def generate_random_prime(min_bits: int, max_bits: int) -> int:
    """
    Generate a random prime number with a specified bit length.
    Uses sympy's randprime which is deterministic and efficient.
    
    Args:
        min_bits: Minimum number of bits
        max_bits: Maximum number of bits
    
    Returns:
        A random prime number in the specified bit range
    """
    # Generate bounds for this bit length
    # For n-bit number: 2^(n-1) <= number < 2^n
    lower_bound = 2**(min_bits - 1)
    upper_bound = 2**max_bits
    
    # Use sympy's randprime to generate a prime in this range
    # randprime(a, b) returns a random prime in [a, b)
    return int(randprime(lower_bound, upper_bound))


def number_to_binary_sequence(num: int, seq_len: int) -> np.ndarray:
    """Convert a number to a binary sequence (most significant bit first).
    
    Returns array of vocab IDs: 0 for PAD, 1 for binary 0, 2 for binary 1.
    """
    # Convert to binary string (without '0b' prefix)
    binary_str = bin(num)[2:]
    
    # Pad with zeros if needed
    if len(binary_str) > seq_len:
        raise ValueError(f"Number {num} requires more than {seq_len} bits (has {len(binary_str)} bits)")
    
    binary_str = binary_str.zfill(seq_len)
    
    # Convert to vocab IDs: '0' -> 1, '1' -> 2
    binary_array = np.array([int(b) + 1 for b in binary_str], dtype=np.uint8)
    
    return binary_array


def generate_semiprimes(num_samples: int, min_bits: int, max_bits: int, seed: int):
    """Generate semiprimes (products of two primes) and their smallest factors.
    
    Args:
        num_samples: Number of semiprimes to generate
        min_bits: Minimum number of bits for each prime factor
        max_bits: Maximum number of bits for each prime factor
        seed: Random seed for reproducibility
    
    Returns:
        List of (semiprime, smaller_factor) tuples
    """
    random.seed(seed)
    np.random.seed(seed)
    
    semiprimes = []
    
    print(f"Generating {num_samples} semiprimes with primes in range [{min_bits}, {max_bits}] bits...")
    for _ in tqdm(range(num_samples)):
        # Generate two random primes
        p1 = generate_random_prime(min_bits, max_bits)
        p2 = generate_random_prime(min_bits, max_bits)
        
        # Calculate the semiprime
        semiprime = p1 * p2
        
        # The smaller factor is the output
        smaller_factor = min(p1, p2)
        
        semiprimes.append((semiprime, smaller_factor))
    
    return semiprimes


def convert_subset(set_name: str, config: DataProcessConfig, num_samples: int, seed_offset: int = 0):
    """Convert a subset (train/test) to the dataset format."""
    
    # Generate semiprimes
    semiprimes = generate_semiprimes(
        num_samples=num_samples,
        min_bits=config.min_bits,
        max_bits=config.max_bits,
        seed=config.seed + seed_offset
    )
    
    # Generate dataset
    results = {k: [] for k in ["inputs", "labels", "puzzle_identifiers", "puzzle_indices", "group_indices"]}
    puzzle_id = 0
    example_id = 0
    
    results["puzzle_indices"].append(0)
    results["group_indices"].append(0)
    
    print(f"Converting {num_samples} examples to dataset format...")
    for semiprime, smaller_factor in tqdm(semiprimes):
        # Convert to binary sequences
        input_seq = number_to_binary_sequence(semiprime, seq_len=config.max_seq_len)
        label_seq = number_to_binary_sequence(smaller_factor, seq_len=config.max_seq_len)
        
        results["inputs"].append(input_seq)
        results["labels"].append(label_seq)
        example_id += 1
        puzzle_id += 1
        
        results["puzzle_indices"].append(example_id)
        results["puzzle_identifiers"].append(0)
        
        # Push group (one example per group)
        results["group_indices"].append(puzzle_id)
    
    # Convert to numpy arrays
    results = {
        "inputs": np.array(results["inputs"], dtype=np.uint8),
        "labels": np.array(results["labels"], dtype=np.uint8),
        
        "group_indices": np.array(results["group_indices"], dtype=np.int32),
        "puzzle_indices": np.array(results["puzzle_indices"], dtype=np.int32),
        "puzzle_identifiers": np.array(results["puzzle_identifiers"], dtype=np.int32),
    }
    
    # Verify vocab range (should be 0, 1, 2)
    assert np.all((results["inputs"] >= 0) & (results["inputs"] <= 2))
    assert np.all((results["labels"] >= 0) & (results["labels"] <= 2))
    
    # Metadata
    metadata = PuzzleDatasetMetadata(
        seq_len=config.max_seq_len,
        vocab_size=3,  # PAD (0) + binary 0 (1) + binary 1 (2)
        pad_id=0,
        ignore_label_id=0,
        blank_identifier_id=0,
        num_puzzle_identifiers=1,
        total_groups=len(results["group_indices"]) - 1,
        mean_puzzle_examples=1.0,
        total_puzzles=len(results["group_indices"]) - 1,
        sets=["all"]
    )
    
    # Save metadata as JSON
    save_dir = os.path.join(config.output_dir, set_name)
    os.makedirs(save_dir, exist_ok=True)
    
    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata.model_dump(), f)
    
    # Save data
    for k, v in results.items():
        np.save(os.path.join(save_dir, f"all__{k}.npy"), v)
    
    # Save IDs mapping (for visualization only)
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>"], f)
    
    print(f"Saved {set_name} dataset to {save_dir}")
    print(f"  - Total examples: {len(results['inputs'])}")
    print(f"  - Input shape: {results['inputs'].shape}")
    print(f"  - Label shape: {results['labels'].shape}")


def generate_factorization_dataset(
    *,
    output_dir: str,
    train_size: int,
    test_size: int = 0,
    min_bits: Optional[int] = None,
    max_bits: int,
    max_seq_len: Optional[int] = None,
    seed: int = 42,
    test_seed_offset: int = 1_000_000,
) -> str:
    """
    Programmatic helper to (re)generate the factorization dataset.

    Args:
        output_dir: Directory that will contain `train/` and `test/` subsets.
        train_size: Number of training examples to create.
        test_size: Number of test examples to create. Set to 0 to skip.
        min_bits: Minimum bit-width for primes. Defaults to `max_bits`.
        max_bits: Maximum bit-width for primes.
        max_seq_len: Sequence length used for binary encodings.
        seed: Base RNG seed for the training split.
        test_seed_offset: Offset added to the seed for the test split to avoid overlap.

    Returns:
        The `output_dir` containing the generated dataset.
    """
    if max_seq_len is None:
        max_seq_len = DataProcessConfig.model_fields["max_seq_len"].default  # type: ignore[index]
    min_bits = min_bits if min_bits is not None else max_bits

    if min_bits > max_bits:
        raise ValueError(
            f"min_bits ({min_bits}) must be <= max_bits ({max_bits}). "
            "Please adjust the dataset_generator settings."
        )

    config = DataProcessConfig(
        output_dir=output_dir,
        train_size=train_size,
        test_size=max(0, test_size),
        min_bits=min_bits,
        max_bits=max_bits,
        max_seq_len=max_seq_len,
        seed=seed,
    )

    convert_subset("train", config, num_samples=config.train_size, seed_offset=0)
    if config.test_size > 0:
        convert_subset("test", config, num_samples=config.test_size, seed_offset=test_seed_offset)

    return output_dir


@cli.command(singleton=True)
def preprocess_data(config: DataProcessConfig):
    """Generate factorization dataset for semiprimes."""
    print("=" * 80)
    print("Generating Factorization Dataset (Semiprimes)")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  - Output directory: {config.output_dir}")
    print(f"  - Train size: {config.train_size}")
    print(f"  - Test size: {config.test_size}")
    print(f"  - Prime bits range: [{config.min_bits}, {config.max_bits}]")
    print(f"  - Seed: {config.seed}")
    print("=" * 80)
    
    convert_subset("train", config, num_samples=config.train_size, seed_offset=0)
    print()
    convert_subset("test", config, num_samples=config.test_size, seed_offset=1000000)


if __name__ == "__main__":
    cli()

