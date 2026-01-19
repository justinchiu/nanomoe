"""Sequence packing utilities for efficient training with variable-length sequences.

Adapted from slime/backends/fsdp_utils/data_packing.py and verl/utils/seqlen_balancing.py
"""

import heapq
import math

import torch

from .types import PackedBatch, Sample


def get_seqlen_balanced_partitions(
    seqlen_list: list[int],
    k_partitions: int,
    equal_size: bool = False,
) -> list[list[int]]:
    """Partition sequences into k groups with balanced total lengths.

    Uses the Karmarkar-Karp differencing algorithm for load balancing.

    Args:
        seqlen_list: Length of each sequence
        k_partitions: Number of partitions to create
        equal_size: If True, each partition must have equal number of items

    Returns:
        List of k partitions, each containing indices into seqlen_list
    """
    if len(seqlen_list) < k_partitions:
        raise ValueError(f"num items ({len(seqlen_list)}) < k_partitions ({k_partitions})")

    class Partition:
        def __init__(self):
            self.total = 0
            self.indices: list[int] = []

        def add(self, idx: int, length: int):
            self.indices.append(idx)
            self.total += length

        def merge(self, other: "Partition"):
            self.indices.extend(other.indices)
            self.total += other.total

        def __lt__(self, other):
            return (self.total, len(self.indices)) < (other.total, len(other.indices))

    # Sort by length descending for greedy assignment
    sorted_items = sorted(enumerate(seqlen_list), key=lambda x: -x[1])

    # Initialize partitions
    partitions = [Partition() for _ in range(k_partitions)]
    heap = [(p.total, i, p) for i, p in enumerate(partitions)]
    heapq.heapify(heap)

    # Greedy assignment: always add to partition with smallest total
    for idx, length in sorted_items:
        _, i, p = heapq.heappop(heap)
        p.add(idx, length)
        heapq.heappush(heap, (p.total, i, p))

    return [sorted(p.indices) for _, _, p in sorted(heap, key=lambda x: x[1])]


def pack_sequences(
    samples: list[Sample],
    max_tokens_per_batch: int | None = None,
    num_packs: int | None = None,
) -> list[PackedBatch]:
    """Pack variable-length sequences into dense batches.

    Args:
        samples: List of samples to pack
        max_tokens_per_batch: Maximum tokens per packed batch
        num_packs: Explicit number of packs (overrides max_tokens_per_batch)

    Returns:
        List of PackedBatch objects ready for training
    """
    if not samples:
        return []

    seq_lengths = [len(s.tokens) for s in samples]

    # Determine number of packs
    if num_packs:
        k_partitions = num_packs
    elif max_tokens_per_batch:
        total_tokens = sum(seq_lengths)
        k_partitions = max(1, math.ceil(total_tokens / max_tokens_per_batch))
    else:
        k_partitions = 1

    # Get balanced partitions
    partitions = get_seqlen_balanced_partitions(seq_lengths, k_partitions)

    # Pack each partition
    result = []
    for indices in partitions:
        cu_seqlens = [0]
        flat_tokens = []
        flat_loss_mask = []
        flat_position_ids = []
        flat_log_probs = []
        flat_advantages = []
        rewards = []

        for i in indices:
            sample = samples[i]
            seq_len = len(sample.tokens)

            flat_tokens.extend(sample.tokens)
            flat_loss_mask.extend(sample.loss_mask)
            flat_position_ids.extend(range(seq_len))
            flat_log_probs.extend(sample.log_probs)

            # Advantage will be filled in during training
            flat_advantages.extend([0.0] * len(sample.log_probs))
            rewards.append(sample.reward)

            cu_seqlens.append(cu_seqlens[-1] + seq_len)

        packed = PackedBatch(
            tokens=torch.tensor(flat_tokens, dtype=torch.long),
            loss_mask=torch.tensor(flat_loss_mask, dtype=torch.int),
            position_ids=torch.tensor(flat_position_ids, dtype=torch.int),
            cu_seqlens=torch.tensor(cu_seqlens, dtype=torch.int32),
            log_probs=torch.tensor(flat_log_probs, dtype=torch.float32),
            advantages=torch.tensor(flat_advantages, dtype=torch.float32),
            rewards=torch.tensor(rewards, dtype=torch.float32),
        )
        result.append(packed)

    return result


def unpack_batch(packed: PackedBatch) -> list[dict]:
    """Unpack a PackedBatch back into individual sequences.

    Useful for debugging or when per-sequence operations are needed.
    """
    cu_seqlens = packed.cu_seqlens.tolist()
    num_seqs = len(cu_seqlens) - 1

    sequences = []
    for i in range(num_seqs):
        start = cu_seqlens[i]
        end = cu_seqlens[i + 1]
        sequences.append(
            {
                "tokens": packed.tokens[start:end],
                "loss_mask": packed.loss_mask[start:end],
                "position_ids": packed.position_ids[start:end],
                "reward": packed.rewards[i],
            }
        )

    return sequences
