"""Deterministic data and selector sampling for distributed PTC-OPD runs.

The data sampler deliberately models a *global microstep*: every rank constructs
the same deterministic epoch permutation, takes the same global batch, and then
takes its rank-local contiguous shard.  Incomplete epoch tails are dropped, so
all ranks always receive exactly ``per_rank_batch`` examples.

The implementation does not depend on Python's process-randomized ``hash``.
Epoch permutations and per-example random-selector seeds are derived with
SHA-256.  Random selector scores are first generated on CPU with a private
``torch.Generator`` and then moved to the requested device.  Consequently, a
sample's scores do not depend on batch order, rank assignment, or global RNG
state.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterator, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor


StableSampleId = Union[str, int]


def _require_integer(name: str, value: int, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("{} must be an integer".format(name))
    if value < minimum:
        raise ValueError("{} must be >= {}".format(name, minimum))
    return value


def _sample_id_bytes(sample_id: StableSampleId) -> bytes:
    """Encode supported IDs without conflating, for example, ``1`` and ``"1"``."""

    if isinstance(sample_id, bool):
        raise TypeError("sample IDs must be strings or integers, not bool")
    if isinstance(sample_id, int):
        return b"integer\x00" + str(sample_id).encode("ascii")
    if isinstance(sample_id, str):
        if not sample_id:
            raise ValueError("sample IDs must not be empty strings")
        return b"string\x00" + sample_id.encode("utf-8")
    raise TypeError("sample IDs must be strings or integers")


def _length_prefix(value: bytes) -> bytes:
    return len(value).to_bytes(8, byteorder="big", signed=False) + value


def _sha256_parts(domain: bytes, *parts: bytes) -> bytes:
    digest = hashlib.sha256()
    digest.update(_length_prefix(domain))
    for part in parts:
        digest.update(_length_prefix(part))
    return digest.digest()


@lru_cache(maxsize=32)
def _epoch_permutation(
    manifest_length: int,
    seed: int,
    epoch: int,
) -> Tuple[int, ...]:
    """Return a platform-independent permutation keyed by seed and epoch."""

    keyed_indices = []
    seed_bytes = str(seed).encode("ascii")
    epoch_bytes = str(epoch).encode("ascii")
    for index in range(manifest_length):
        key = _sha256_parts(
            b"ptc-opd/epoch-permutation/v1",
            seed_bytes,
            epoch_bytes,
            str(index).encode("ascii"),
        )
        keyed_indices.append((key, index))
    keyed_indices.sort()
    return tuple(index for _, index in keyed_indices)


@dataclass(frozen=True)
class BatchAssignment:
    """One rank's immutable assignment for a global training microstep."""

    global_microstep: int
    epoch: int
    microstep_in_epoch: int
    rank: int
    indices: Tuple[int, ...]
    sample_ids: Tuple[StableSampleId, ...]


class DeterministicDistributedBatchSampler:
    """Map global microsteps to deterministic, disjoint rank-local batches.

    Args:
        manifest_length: Number of records in the immutable training manifest.
        seed: Non-negative run seed used only for the epoch permutation.
        rank: Local process's DDP rank in ``[0, world_size)``.
        world_size: Number of DDP ranks participating in this run.
        per_rank_batch: Physical batch size consumed by each rank per microstep.
        sample_ids: Optional stable ID for every manifest row.  When omitted,
            integer manifest indices are used as IDs.  Supplied IDs must be
            unique so keyed selector randomness identifies one record exactly.

    ``global_microstep`` is zero-based and may grow across epochs.  Its epoch is
    derived rather than stored, which makes exact resume independent of a
    stateful DataLoader iterator.
    """

    def __init__(
        self,
        manifest_length: int,
        seed: int,
        rank: int,
        world_size: int,
        per_rank_batch: int,
        sample_ids: Optional[Sequence[StableSampleId]] = None,
    ) -> None:
        self.manifest_length = _require_integer(
            "manifest_length", manifest_length, 1
        )
        self.seed = _require_integer("seed", seed, 0)
        self.world_size = _require_integer("world_size", world_size, 1)
        self.rank = _require_integer("rank", rank, 0)
        self.per_rank_batch = _require_integer(
            "per_rank_batch", per_rank_batch, 1
        )
        if self.rank >= self.world_size:
            raise ValueError("rank must be smaller than world_size")

        self.global_batch_size = self.world_size * self.per_rank_batch
        self.microsteps_per_epoch = self.manifest_length // self.global_batch_size
        self.dropped_per_epoch = (
            self.manifest_length
            - self.microsteps_per_epoch * self.global_batch_size
        )
        if self.microsteps_per_epoch == 0:
            raise ValueError(
                "manifest_length must contain at least one full global batch "
                "when drop_last=True"
            )

        if sample_ids is None:
            normalized_ids: Tuple[StableSampleId, ...] = tuple(
                range(self.manifest_length)
            )
        else:
            if len(sample_ids) != self.manifest_length:
                raise ValueError(
                    "sample_ids length must equal manifest_length"
                )
            normalized_ids = tuple(sample_ids)
            encoded_ids = tuple(_sample_id_bytes(value) for value in normalized_ids)
            if len(set(encoded_ids)) != len(encoded_ids):
                raise ValueError("sample_ids must be unique")
        self.sample_ids = normalized_ids

    def batch(self, global_microstep: int) -> BatchAssignment:
        """Return this rank's batch for ``global_microstep``."""

        global_microstep = _require_integer(
            "global_microstep", global_microstep, 0
        )
        epoch, microstep_in_epoch = divmod(
            global_microstep, self.microsteps_per_epoch
        )
        permutation = _epoch_permutation(
            self.manifest_length, self.seed, epoch
        )

        global_start = microstep_in_epoch * self.global_batch_size
        rank_start = global_start + self.rank * self.per_rank_batch
        rank_stop = rank_start + self.per_rank_batch
        indices = permutation[rank_start:rank_stop]
        if len(indices) != self.per_rank_batch:
            raise RuntimeError("internal error: rank received an incomplete batch")
        ids = tuple(self.sample_ids[index] for index in indices)
        return BatchAssignment(
            global_microstep=global_microstep,
            epoch=epoch,
            microstep_in_epoch=microstep_in_epoch,
            rank=self.rank,
            indices=indices,
            sample_ids=ids,
        )

    def iter_epoch(self, epoch: int) -> Iterator[BatchAssignment]:
        """Yield this rank's assignments for one epoch."""

        epoch = _require_integer("epoch", epoch, 0)
        first_global_microstep = epoch * self.microsteps_per_epoch
        for offset in range(self.microsteps_per_epoch):
            yield self.batch(first_global_microstep + offset)


def keyed_random_scores(
    sample_ids: Sequence[StableSampleId],
    num_codebooks: int,
    num_timesteps: int,
    run_seed: int,
    optimizer_step: int,
    *,
    random_namespace: int = 5701,
    device: Union[str, torch.device] = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Return deterministic ``[B, Q, T]`` random selector scores.

    A private CPU generator is seeded independently for every stable sample ID
    using SHA-256 over ``(random_namespace, run_seed, optimizer_step,
    sample_id, codebook_id)``.  The frozen protocol uses namespace ``5701``;
    exposing it explicitly makes the recorded seed ledger auditable.
    Batch position, accumulation microstep, and DDP rank are intentionally
    absent from the key.  Therefore moving an example within the same optimizer
    update cannot change its matched-random control mask.  The function neither
    reads nor mutates PyTorch's global RNG state.

    Scores are generated in float32 before the requested cast.  Using float32
    for random ranking is recommended because lower-precision casts can create
    ties.
    """

    num_codebooks = _require_integer("num_codebooks", num_codebooks, 1)
    num_timesteps = _require_integer("num_timesteps", num_timesteps, 1)
    run_seed = _require_integer("run_seed", run_seed, 0)
    optimizer_step = _require_integer("optimizer_step", optimizer_step, 0)
    random_namespace = _require_integer(
        "random_namespace", random_namespace, 0
    )
    if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
        raise TypeError("dtype must be a floating-point torch.dtype")

    device_object = torch.device(device)
    rows = []
    run_seed_bytes = str(run_seed).encode("ascii")
    optimizer_step_bytes = str(optimizer_step).encode("ascii")
    namespace_bytes = str(random_namespace).encode("ascii")
    for sample_id in sample_ids:
        codebook_rows = []
        for codebook_id in range(num_codebooks):
            digest = _sha256_parts(
                b"ptc-opd/keyed-random-scores/v1",
                namespace_bytes,
                run_seed_bytes,
                optimizer_step_bytes,
                _sample_id_bytes(sample_id),
                str(codebook_id).encode("ascii"),
            )
            # Keep the seed within the non-negative signed 64-bit range accepted
            # by every supported PyTorch release.
            generator_seed = int.from_bytes(
                digest[:8], byteorder="big", signed=False
            ) & ((1 << 63) - 1)
            generator = torch.Generator(device="cpu")
            generator.manual_seed(generator_seed)
            codebook_rows.append(
                torch.rand(
                    (num_timesteps,),
                    generator=generator,
                    device="cpu",
                    dtype=torch.float32,
                )
            )
        rows.append(torch.stack(codebook_rows, dim=0))

    if rows:
        scores = torch.stack(rows, dim=0)
    else:
        scores = torch.empty(
            (0, num_codebooks, num_timesteps),
            device="cpu",
            dtype=torch.float32,
        )
    return scores.to(device=device_object, dtype=dtype)


__all__ = [
    "BatchAssignment",
    "DeterministicDistributedBatchSampler",
    "StableSampleId",
    "keyed_random_scores",
]
