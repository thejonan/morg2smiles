"""Evaluation: generate, judge, score.

One function, used by both the training loop (on validation, every epoch) and
the experiment runner (on whichever split was asked for). Keeping it in one
place means the number the iteration loop chases is computed by exactly the
same code as the number that gets reported.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import replace

import torch
from tqdm import tqdm

from .fingerprints import FPConfig, compute
from .generate import beam_search, sample
from .metrics import QueryOutcome, evaluate
from .model import Morg2SmilesModel
from .noise import NoiseConfig, clean_probabilities, query_rng, simulate
from .oracle import OracleResult, check, check_many, with_log_likelihood

__all__ = ["evaluate_model", "generate_outcomes"]


def generate_outcomes(
    model: Morg2SmilesModel,
    smiles: Sequence[str],
    *,
    k: int = 20,
    fp_unseen: Sequence[bool] | None = None,
    strategy: str = "sample",
    batch_size: int = 16,
    temperature: float = 1.0,
    top_p: float = 1.0,
    budget: str = "molecules",
    oversample: float = 2.0,
    max_oversample: float = 10.0,
    seed: int | None = 0,
    noise: NoiseConfig | None = None,
    noise_seed: int = 0,
    show_progress: bool = True,
    **kwargs,
) -> list[QueryOutcome]:
    """Generate k candidates per molecule and judge them against its fingerprint.

    Args:
        model: A trained model.
        smiles: Reference molecules. Their fingerprints are the queries, and
            they double as the ``exact_structure`` reference.
        k: Candidates per query.
        fp_unseen: Per-molecule flags from
            :func:`morg2smiles.data.dataset.fp_unseen_mask`. When omitted, every
            query is treated as seen, which makes the ``fp_unseen`` slice empty
            and the primary metric zero -- deliberately, so a missing mask is
            obvious rather than quietly flattering.
        strategy: ``"sample"`` or ``"beam"``.
        batch_size: Queries per device batch. Sampling runs k per query, so the
            real batch on the device is ``batch_size * k``. The default was
            measured on an M2: throughput is flat from 16 to 32 and degrades
            sharply at 64, where the cache stops fitting comfortably.
        budget: ``"molecules"`` or ``"strings"``; see
            :meth:`morg2smiles.metrics.QueryOutcome.scored`.
        oversample: The *first* draw is ``k * oversample`` strings per query.
            Queries still short of k distinct candidates are then topped up,
            k strings at a time, until they fill or hit ``max_oversample``.
        max_oversample: Cap on raw strings per query, as a multiple of k. Set
            it equal to ``oversample`` to reproduce the old fixed-draw numbers.

            Why adaptive rather than a bigger constant: 2.0 was calibrated on
            the 100k model, where 40 strings gave ~21 distinct molecules. The
            1M model repeats itself far more (duplicate rate 0.569 against
            0.247) and got only 12.6 from the same 40, so its recovery@20 was
            measured over ~13 candidates while the weaker model had 20. Better
            models explore less, so any fixed multiplier short-changes exactly
            the model that matters most. Topping up per query removes the
            constant; ``budget_filled_rate`` in the report says whether the cap
            was ever reached.
        seed: Seed for the sampling RNG, so an evaluation is reproducible.
            Defaults to 0 rather than None because unseeded sampling made
            repeated evaluations of *the same checkpoint* differ by 2.2 points
            of recovery@20 (0.5729 against 0.5508, n=995) -- roughly 1.4
            standard errors, and easily enough to invent or hide a difference
            between two training runs. Pass None for genuinely fresh samples.
        noise: Probability models only. The simulated predictor whose output
            the model is queried with; None queries with the exact fingerprint
            at p = 1. Use :meth:`NoiseConfig.for_evaluation`. Candidates are
            still judged against the *true* fingerprint, so recovery@k keeps
            its meaning, and each also gets a ``log_likelihood`` under the
            noisy query, for the orders a caller without an oracle would use.
        noise_seed: Keys the simulated prediction together with the molecule,
            so one molecule gets the same noisy query in every evaluation.

    Returns:
        One :class:`QueryOutcome` per input molecule, in input order.
    """
    fp_cfg: FPConfig = model.fp_cfg
    if fp_unseen is not None and len(fp_unseen) != len(smiles):
        raise ValueError(f"fp_unseen has {len(fp_unseen)} entries for {len(smiles)} molecules")
    if budget not in ("molecules", "strings"):
        raise ValueError(f"budget must be 'molecules' or 'strings', got {budget!r}")

    fps = [compute(s, fp_cfg) for s in smiles]
    probabilistic = model.cfg.probabilistic
    if noise is not None and not probabilistic:
        raise ValueError("noise needs a probability-input model (fp_input: probability)")
    if probabilistic:
        # What the model is asked: a simulated prediction, or the exact bits.
        queries = [
            simulate(fp, fp_cfg.n_bits, noise, query_rng(s, noise_seed))
            if noise is not None
            else clean_probabilities(fp)
            for s, fp in zip(smiles, fps, strict=True)
        ]
    else:
        queries = fps
    outcomes: list[QueryOutcome] = []
    n_draw = max(k, int(round(k * oversample)))
    max_draw = max(n_draw, int(round(k * max_oversample)))

    def score(index: int, r: OracleResult) -> OracleResult:
        # Probability mode: how probable the candidate is under the query.
        if not probabilistic or r.duplicate:
            return r
        return with_log_likelihood(r, queries[index], fp_cfg)

    def judge(index: int, candidates: Sequence[str]) -> QueryOutcome:
        results = check_many(candidates, fps[index], fp_cfg, reference_smiles=smiles[index])
        return QueryOutcome(
            results=[score(index, r) for r in results],
            fp_unseen=bool(fp_unseen[index]) if fp_unseen is not None else False,
            reference_smiles=smiles[index],
            budget=budget,
            n_samples=len(candidates),
        )

    def fills(results: Sequence[OracleResult]) -> int:
        # What QueryOutcome.scored() will count, without building one.
        if budget == "molecules":
            return sum(1 for r in results if r.valid and not r.duplicate)
        return sum(1 for r in results if not r.duplicate)

    if strategy == "beam":
        # Beam search already returns distinct sequences, so oversampling it
        # would only add lower-ranked beams; k stays k.
        iterator = tqdm(range(len(smiles)), desc="beam", unit=" mol", disable=not show_progress)
        for i in iterator:
            outcomes.append(judge(i, beam_search(model, queries[i], k=k, **kwargs)))
        return outcomes

    # Top-ups draw from their own stream. Sharing one would let a top-up in
    # batch 1 shift every later batch's first draw, so recovery@1 would move by
    # seed noise whenever the cap changed. Separate, the first draw is identical
    # to the fixed-oversample protocol and only the extra candidates are new.
    generator = topup_generator = None
    if seed is not None:
        device = next(model.parameters()).device
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        topup_generator = torch.Generator(device=device)
        topup_generator.manual_seed(seed + 1)

    total = (len(smiles) + batch_size - 1) // batch_size
    iterator = tqdm(
        range(0, len(smiles), batch_size),
        total=total,
        desc=f"sampling k={k}",
        unit=" batch",
        disable=not show_progress,
    )
    for start in iterator:
        chunk = list(range(start, min(start + batch_size, len(smiles))))
        # Judged incrementally, with check_many's dedupe rules, so a top-up
        # round only pays the oracle for its new strings.
        results: dict[int, list[OracleResult]] = {i: [] for i in chunk}
        seen_raw: dict[int, set[str]] = {i: set() for i in chunk}
        seen_canonical: dict[int, set[str]] = {i: set() for i in chunk}
        drawn = dict.fromkeys(chunk, 0)

        pending, per_query, stream = chunk, n_draw, generator
        while pending:
            batch_candidates = sample(
                model,
                [queries[i] for i in pending],
                k=per_query,
                temperature=temperature,
                top_p=top_p,
                generator=stream,
                **kwargs,
            )
            for i, candidates in zip(pending, batch_candidates, strict=True):
                drawn[i] += len(candidates)
                for candidate in candidates:
                    if candidate in seen_raw[i]:
                        continue
                    seen_raw[i].add(candidate)
                    r = check(candidate, fps[i], fp_cfg, reference_smiles=smiles[i])
                    if r.canonical is not None:
                        if r.canonical in seen_canonical[i]:
                            r = replace(r, duplicate=True)
                        else:
                            seen_canonical[i].add(r.canonical)
                    results[i].append(score(i, r))
            per_query, stream = k, topup_generator
            pending = [i for i in pending if fills(results[i]) < k and drawn[i] + k <= max_draw]

        for i in chunk:
            outcomes.append(
                QueryOutcome(
                    results=results[i],
                    fp_unseen=bool(fp_unseen[i]) if fp_unseen is not None else False,
                    reference_smiles=smiles[i],
                    budget=budget,
                    n_samples=drawn[i],
                )
            )
    return outcomes


def evaluate_model(
    model: Morg2SmilesModel,
    smiles: Sequence[str],
    *,
    k: int = 20,
    ks: Sequence[int] | None = None,
    fp_unseen: Sequence[bool] | None = None,
    **kwargs,
) -> dict:
    """Generate, judge and score, returning a full metric report.

    Args:
        k: Candidates to generate per query. This is the budget.
        ks: The k values to report recovery at. Defaults to the powers of the
            budget that fit -- reporting recovery@100 from 20 candidates would
            just restate recovery@20.

    Returns:
        The report from :func:`morg2smiles.metrics.evaluate`.
    """
    ks = ks or [n for n in (1, 5, 20, 100) if n <= k] or [k]
    started = time.perf_counter()
    noise = kwargs.pop("noise", None)
    outcomes = generate_outcomes(model, smiles, k=k, fp_unseen=fp_unseen, noise=noise, **kwargs)
    elapsed = time.perf_counter() - started
    report = evaluate(outcomes, ks=ks, elapsed_seconds=elapsed)
    report["generation"] = {
        "k": k,
        **{
            key: value
            for key, value in kwargs.items()
            if isinstance(value, (int, float, str, bool))
        },
    }
    if model.cfg.probabilistic:
        # What the model was queried with; None means the exact fingerprint.
        report["input_noise"] = noise.to_dict() if noise is not None else None
    return report
