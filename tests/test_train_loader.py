"""Tests for the training DataLoader's handling of epochs and worker processes.

With ``num_workers > 0`` the loader used to keep its workers alive between
epochs. Each worker holds its own copy of the dataset, so ``set_epoch()`` never
reached them, and every epoch replayed epoch 0's random SMILES. These pin the
two properties that matter: workers see every epoch, and ``num_workers``
changes only speed -- the batches are identical to building them in the main
process.
"""

from __future__ import annotations

import random

import torch
from torch.utils.data import Dataset

from morg2smiles.train import TrainConfig, _train_loader


class _EpochEcho(Dataset):
    """Shaped like FingerprintDataset's items, and seeded the same way.

    ``tokens`` carries the epoch this copy of the dataset believes it is in and
    a draw from an RNG seeded on (seed, epoch, index), as the real dataset seeds
    its SMILES randomisation.
    """

    def __init__(self, n: int = 24, seed: int = 0):
        self.n, self.seed, self.epoch = n, seed, 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> dict:
        draw = random.Random((self.seed, self.epoch, i).__hash__()).randrange(10**6)
        return {
            "fp_indices": torch.tensor([i]),
            "fp_counts": torch.tensor([1]),
            "tokens": torch.tensor([self.epoch, draw]),
            "index": i,
        }


def _epochs(num_workers: int, n_epochs: int = 3) -> list[list[tuple[int, int, int]]]:
    """(index, epoch, draw) for every item, in batch order, per epoch."""
    dataset = _EpochEcho()
    cfg = TrainConfig(batch_size=8, num_workers=num_workers)
    loader = _train_loader(dataset, cfg, sampler=None, pad_id=0)
    out = []
    for epoch in range(n_epochs):
        dataset.set_epoch(epoch)  # exactly what train() does
        out.append(
            [
                (int(i), int(t[0]), int(t[1]))
                for batch in loader
                for i, t in zip(batch["index"], batch["tokens"], strict=True)
            ]
        )
    return out


def test_workers_see_every_epoch():
    seen = [{epoch for _, epoch, _ in items} for items in _epochs(num_workers=2)]
    assert seen == [{0}, {1}, {2}]


def test_workers_build_the_same_batches_as_the_main_process():
    torch.manual_seed(0)
    in_process = _epochs(num_workers=0)
    torch.manual_seed(0)
    with_workers = _epochs(num_workers=2)
    assert with_workers == in_process
    # And the shuffle really moves between epochs, so the comparison has teeth.
    assert [i for i, _, _ in in_process[0]] != [i for i, _, _ in in_process[1]]
