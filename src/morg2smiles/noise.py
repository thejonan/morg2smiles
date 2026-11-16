"""Probabilistic fingerprints: how the model reads them, and how to fake them.

A fingerprint predicted from a spectrum is not a set of bits but a probability
per bit, and it is wrong in two directions: some of the molecule's real bits
get low probability (misses), and some bits it does not have get high
probability (false positives). A model trained only on exact fingerprints has
never seen either, so it has to be trained on fingerprints that look like a
predictor's output. This module supplies the three pieces that takes.

**Reading probabilities** (:class:`ProbInput`). The set encoder already embeds
each set bit as ``bit_embedding[bit] + count_embedding[count]``. In probability
mode the count is replaced by a *probability level*: ``p`` cut into
``n_bins`` equal bins, level 1 to ``n_bins``, with 0 left for padding as
before. Only the ``max_bits`` most probable bits with ``p >= min_prob`` become
tokens; a bit left out reads as "probably absent". A clean fingerprint is the
special case where every real bit has ``p = 1`` and lands in the top level.

**Faking a predictor** (:class:`NoiseConfig`, :func:`simulate`). Starting from
a molecule's true bits: each is *missed* with ``miss_rate`` and then gets a low
probability, otherwise a high one; then about ``false_bits`` bits the molecule
does not have are added with their own probabilities. The defaults are
placeholders. The noise that matters is your predictor's, so measure its
per-bit miss and false-positive rates on its own validation set and pass them
as ``rates_file``; see :class:`NoiseConfig`.

**Ranking without an oracle** (:func:`log_likelihood`). With a probabilistic
query there is no exact fingerprint to match, so candidates are scored by how
probable their own fingerprint is under the predicted probabilities.

Pure numpy, no torch and no RDKit, so the dataset workers and the tests can
use it cheaply.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from pathlib import Path

import numpy as np

__all__ = [
    "ProbInput",
    "NoiseConfig",
    "simulate",
    "clean_probabilities",
    "query_rng",
    "log_likelihood",
    "load_rates",
]

#: Probabilities are clipped to ``[EPS, 1 - EPS]`` before taking logs, so one
#: confidently wrong bit costs a large but finite penalty instead of -inf.
EPS = 1e-3


@dataclass(frozen=True)
class ProbInput:
    """How a probability fingerprint becomes encoder tokens.

    Lives in :class:`~morg2smiles.model.ModelConfig` (as ``prob_bins``,
    ``prob_max_bits`` and ``prob_min``), because a model only works on inputs
    encoded exactly as in training, and the checkpoint must say how that was.

    Attributes:
        n_bins: Probability levels. 16 resolves 0.06 steps, finer than any
            predictor is calibrated to.
        max_bits: Most tokens fed to the encoder. Encoder cost grows with this,
            so it is the main cost knob; 128 leaves room for ~50 real bits plus
            a predictor's plausible false positives.
        min_prob: Bits below this are dropped before the cap applies.
    """

    n_bins: int = 16
    max_bits: int = 128
    min_prob: float = 0.05

    def __post_init__(self) -> None:
        if self.n_bins < 1:
            raise ValueError(f"n_bins must be >= 1, got {self.n_bins}")
        if self.max_bits < 1:
            raise ValueError(f"max_bits must be >= 1, got {self.max_bits}")
        if not 0.0 <= self.min_prob <= 1.0:
            raise ValueError(f"min_prob must be in [0, 1], got {self.min_prob}")

    def level(self, p: float) -> int:
        """Embedding index for probability ``p``: 1 to ``n_bins``."""
        return 1 + min(self.n_bins - 1, int(p * self.n_bins))

    def levels(self, probs: Mapping[int, float]) -> tuple[list[int], list[int]]:
        """Select and encode bits as parallel ``(bit indices, levels)`` lists.

        Keeps the ``max_bits`` most probable bits at or above ``min_prob``,
        ties broken by bit index so the result is deterministic. If nothing
        clears ``min_prob``, the single most probable bit is kept anyway: an
        empty input would leave the encoder nothing to attend to.

        Raises:
            ValueError: If a probability lies outside ``[0, 1]``.
        """
        for bit, p in probs.items():
            if not 0.0 <= p <= 1.0 or math.isnan(p):
                raise ValueError(f"bit {bit} has probability {p}, outside [0, 1]")
        ranked = sorted(probs.items(), key=lambda item: (-item[1], item[0]))
        kept = [(bit, p) for bit, p in ranked if p >= self.min_prob and p > 0.0]
        kept = kept[: self.max_bits] or ranked[:1]
        kept.sort()
        return [int(bit) for bit, _ in kept], [self.level(p) for _, p in kept]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class NoiseConfig:
    """A synthetic fingerprint predictor, for training and evaluation.

    Set under ``train.noise`` in a config. Beta distributions are given as
    ``[a, b]``; the mean of Beta(a, b) is a / (a + b).

    Attributes:
        clean_fraction: Share of training samples left exact (every real bit
            at p = 1), so the model keeps its exact-recovery skill. Evaluation
            always uses 0; see :meth:`for_evaluation`.
        miss_rate: Chance that a real bit is missed. Overridden per bit by
            ``rates_file``.
        false_bits: Mean number of spurious bits per molecule (Poisson).
            Overridden by ``rates_file``.
        hit_conf: Probability given to a real bit that was not missed.
        miss_conf: Probability given to a missed real bit. Often below
            ``ProbInput.min_prob``, so the bit simply disappears.
        false_conf: Probability given to a spurious bit.
        rates_file: Optional JSON with per-bit rates measured on the real
            predictor: ``{"miss_rate": [...], "false_rate": [...]}``, one entry
            per bit. ``miss_rate[i]`` is P(predictor misses bit i | the
            molecule has it); ``false_rate[i]`` is P(predictor reports bit i |
            the molecule lacks it), using the same threshold as "reports".
            Each list must have ``n_bits`` entries.
    """

    clean_fraction: float = 0.2
    miss_rate: float = 0.15
    false_bits: float = 8.0
    hit_conf: tuple[float, float] = (8.0, 2.0)
    miss_conf: tuple[float, float] = (2.0, 8.0)
    false_conf: tuple[float, float] = (2.0, 5.0)
    rates_file: str | None = None

    def __post_init__(self) -> None:
        for name in ("clean_fraction", "miss_rate"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        if self.false_bits < 0:
            raise ValueError(f"false_bits must be >= 0, got {self.false_bits}")
        for name in ("hit_conf", "miss_conf", "false_conf"):
            value = tuple(float(x) for x in getattr(self, name))
            if len(value) != 2 or min(value) <= 0:
                raise ValueError(f"{name} must be two positive Beta parameters, got {value}")
            object.__setattr__(self, name, value)

    def for_evaluation(self) -> NoiseConfig:
        """The same predictor with no clean samples: evaluation is all noisy."""
        return replace(self, clean_fraction=0.0)

    def to_dict(self) -> dict:
        d = asdict(self)
        for name in ("hit_conf", "miss_conf", "false_conf"):
            d[name] = list(d[name])
        return d

    @classmethod
    def from_dict(cls, d: Mapping) -> NoiseConfig:
        known = set(cls.__dataclass_fields__)
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown noise settings: {sorted(unknown)}")
        return cls(**dict(d))


@lru_cache(maxsize=4)
def load_rates(path: str, n_bits: int) -> tuple[np.ndarray, np.ndarray]:
    """Read and check a per-bit rates file. Cached, so workers read it once."""
    payload = json.loads(Path(path).read_text())
    miss = np.asarray(payload["miss_rate"], dtype=np.float64)
    false = np.asarray(payload["false_rate"], dtype=np.float64)
    for name, arr in (("miss_rate", miss), ("false_rate", false)):
        if arr.shape != (n_bits,):
            raise ValueError(f"{path}: {name} has {arr.size} entries, expected {n_bits}")
        if arr.min() < 0 or arr.max() > 1:
            raise ValueError(f"{path}: {name} values must lie in [0, 1]")
    return miss, false


def clean_probabilities(bits: Iterable[int]) -> dict[int, float]:
    """An exact fingerprint in probability form: every real bit at p = 1."""
    return dict.fromkeys((int(b) for b in bits), 1.0)


def simulate(
    bits: Iterable[int], n_bits: int, cfg: NoiseConfig, rng: np.random.Generator
) -> dict[int, float]:
    """Turn a molecule's true bits into a plausible predictor output.

    Args:
        bits: The molecule's set bits (a sparse fingerprint's keys).
        n_bits: Fingerprint width.
        cfg: The predictor's error profile.
        rng: Source of randomness; seed it per sample for reproducibility.

    Returns:
        Bit -> probability, for the real bits and the spurious ones. Bits not
        listed have probability 0.
    """
    true = np.fromiter((int(b) for b in bits), dtype=np.int64)
    if rng.random() < cfg.clean_fraction:
        return clean_probabilities(true)

    rates = load_rates(cfg.rates_file, n_bits) if cfg.rates_file else None
    miss_p = rates[0][true] if rates is not None else cfg.miss_rate
    missed = rng.random(true.size) < miss_p
    p_true = np.where(
        missed,
        rng.beta(*cfg.miss_conf, size=true.size),
        rng.beta(*cfg.hit_conf, size=true.size),
    )

    if rates is not None:
        spurious = np.flatnonzero(rng.random(n_bits) < rates[1])
        spurious = spurious[~np.isin(spurious, true)]
    else:
        n_false = min(int(rng.poisson(cfg.false_bits)), n_bits - true.size)
        if n_false > 0:
            # Drawn in random order, so taking the first n_false after removing
            # the real bits is still uniform over the bits the molecule lacks.
            pool = rng.choice(n_bits, size=min(n_bits, n_false + true.size), replace=False)
            spurious = pool[~np.isin(pool, true)][:n_false]
        else:
            spurious = np.empty(0, dtype=np.int64)
    p_false = rng.beta(*cfg.false_conf, size=spurious.size)

    probs = {int(b): float(p) for b, p in zip(true, p_true, strict=True)}
    probs.update({int(b): float(p) for b, p in zip(spurious, p_false, strict=True)})
    return probs


def query_rng(smiles: str, seed: int = 0) -> np.random.Generator:
    """A generator keyed on the molecule itself, for evaluation noise.

    Keyed on the SMILES rather than the position in a list, so a molecule gets
    the same simulated prediction in the in-training subsample, in the final
    evaluation and in every model's evaluation -- the protocol stays fixed
    while the models change.
    """
    digest = hashlib.blake2b(smiles.encode(), digest_size=8).digest()
    return np.random.default_rng([int(seed), int.from_bytes(digest, "little")])


def log_likelihood(candidate_bits: Iterable[int], probs: Mapping[int, float], n_bits: int) -> float:
    """Log-probability of a candidate's fingerprint under predicted bit probabilities.

    Treats bits as independent Bernoullis: sum over every bit of
    ``b log p + (1 - b) log(1 - p)``, where ``b`` is whether the candidate sets
    the bit. Bits missing from ``probs`` count as probability 0, clipped to
    :data:`EPS` like every other value, so a candidate pays for each bit it
    sets that the predictor never mentioned.

    Higher is better. Comparable across candidates for one query, not across
    queries.
    """
    candidate = {int(b) for b in candidate_bits}
    total = 0.0
    for bit, p in probs.items():
        p = min(max(float(p), EPS), 1.0 - EPS)
        total += math.log(p) if bit in candidate else math.log1p(-p)
    outside_set = sum(1 for b in candidate if b not in probs)
    outside_unset = n_bits - len(probs) - outside_set
    return total + outside_set * math.log(EPS) + outside_unset * math.log1p(-EPS)
