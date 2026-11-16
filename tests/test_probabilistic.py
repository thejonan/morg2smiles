"""Tests for probability-input models: the noise simulator, the encoding, and
the paths from training through evaluation and generation."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch

from morg2smiles.data.dataset import FingerprintDataset, collate
from morg2smiles.evaluation import evaluate_model
from morg2smiles.fingerprints import FPConfig, compute
from morg2smiles.generate import Morg2Smiles, _as_sparse
from morg2smiles.metrics import QueryOutcome, recovery_at_k
from morg2smiles.model import ModelConfig, Morg2SmilesModel, build_model
from morg2smiles.noise import (
    EPS,
    NoiseConfig,
    ProbInput,
    clean_probabilities,
    log_likelihood,
    query_rng,
    simulate,
)
from morg2smiles.oracle import OracleResult, with_log_likelihood
from morg2smiles.tokenizer import SmilesTokenizer

MOLECULES = [
    "CC(=O)Oc1ccccc1C(=O)O",
    "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
    "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",
    "CCO",
    "c1ccccc1",
    "O=S(=O)(N)c1ccccc1",
]
FP4096 = FPConfig(n_bits=4096)
PROB = ModelConfig(
    d_model=64,
    n_heads=4,
    d_ff=128,
    n_encoder_layers=2,
    n_decoder_layers=2,
    max_tokens=48,
    fp_input="probability",
)
NOISE = NoiseConfig(clean_fraction=0.0)


@pytest.fixture
def tokenizer():
    return SmilesTokenizer.build(MOLECULES)


@pytest.fixture
def model(tokenizer):
    torch.manual_seed(0)
    return build_model(PROB, FP4096, tokenizer).eval()


# -- encoding ------------------------------------------------------------------
def test_levels_bin_probabilities_and_reserve_zero_for_padding():
    enc = ProbInput(n_bins=16)
    assert enc.level(0.0) == 1
    assert enc.level(0.5) == 9
    assert enc.level(1.0) == 16
    assert min(enc.level(p) for p in np.linspace(0, 1, 101)) >= 1


def test_levels_keep_the_most_probable_bits_above_the_floor():
    enc = ProbInput(n_bins=4, max_bits=2, min_prob=0.1)
    indices, levels = enc.levels({5: 0.9, 1: 0.05, 9: 0.5, 3: 0.95})
    assert indices == [3, 5]  # 9 loses the cap, 1 is under the floor; sorted by bit
    assert levels == [enc.level(0.95), enc.level(0.9)]


def test_levels_never_return_an_empty_input():
    indices, _ = ProbInput(min_prob=0.5).levels({7: 0.2, 2: 0.1})
    assert indices == [7]


def test_levels_reject_values_that_are_not_probabilities():
    with pytest.raises(ValueError, match="outside"):
        ProbInput().levels({1: 1.5})


def test_clean_fingerprint_sits_in_the_top_level():
    _, levels = ProbInput().levels(clean_probabilities([3, 8, 40]))
    assert levels == [16, 16, 16]


# -- noise ---------------------------------------------------------------------
def test_simulation_is_reproducible_and_plausible():
    bits = list(compute(MOLECULES[0], FP4096))
    a = simulate(bits, 4096, NOISE, np.random.default_rng(1))
    b = simulate(bits, 4096, NOISE, np.random.default_rng(1))
    assert a == b
    assert all(0.0 <= p <= 1.0 for p in a.values())
    assert set(bits) <= set(a)  # every real bit gets a probability, missed or not
    assert all(0 <= bit < 4096 for bit in a)


def test_simulation_rates_on_average():
    """Misses and spurious bits arrive at the configured rates."""
    cfg = NoiseConfig(clean_fraction=0.0, miss_rate=0.3, false_bits=10.0)
    bits = list(range(0, 400, 8))  # 50 real bits
    rng = np.random.default_rng(0)
    low, spurious, n = 0, 0, 400
    for _ in range(n):
        probs = simulate(bits, 4096, cfg, rng)
        low += sum(probs[b] < 0.5 for b in bits)
        spurious += len(probs) - len(bits)
    # Missed bits are drawn from Beta(2, 8) and kept ones from Beta(8, 2), so
    # the share below 0.5 is close to, not exactly, the miss rate.
    assert 0.25 < low / (n * len(bits)) < 0.35
    assert 9.0 < spurious / n < 11.0


def test_clean_fraction_one_returns_the_exact_fingerprint():
    probs = simulate([1, 2, 3], 4096, NoiseConfig(clean_fraction=1.0), np.random.default_rng(0))
    assert probs == {1: 1.0, 2: 1.0, 3: 1.0}


def test_per_bit_rates_file(tmp_path):
    n_bits = 64
    path = tmp_path / "rates.json"
    false_rate = [0.0] * n_bits
    false_rate[10] = 1.0
    path.write_text(json.dumps({"miss_rate": [0.0] * n_bits, "false_rate": false_rate}))
    cfg = NoiseConfig(clean_fraction=0.0, rates_file=str(path))
    probs = simulate([1, 2], n_bits, cfg, np.random.default_rng(0))
    assert set(probs) == {1, 2, 10}


def test_rates_file_must_match_the_width(tmp_path):
    path = tmp_path / "rates.json"
    path.write_text(json.dumps({"miss_rate": [0.1] * 8, "false_rate": [0.1] * 8}))
    cfg = NoiseConfig(rates_file=str(path))
    with pytest.raises(ValueError, match="expected 4096"):
        simulate([1], 4096, cfg, np.random.default_rng(0))


def test_noise_config_round_trips_and_rejects_typos():
    cfg = NoiseConfig(miss_rate=0.2, hit_conf=(5, 1))
    assert NoiseConfig.from_dict(json.loads(json.dumps(cfg.to_dict()))) == cfg
    assert cfg.for_evaluation().clean_fraction == 0.0
    with pytest.raises(ValueError, match="unknown"):
        NoiseConfig.from_dict({"miss_rat": 0.1})


def test_query_noise_is_keyed_on_the_molecule():
    bits = list(compute(MOLECULES[0], FP4096))
    one = simulate(bits, 4096, NOISE, query_rng(MOLECULES[0]))
    again = simulate(bits, 4096, NOISE, query_rng(MOLECULES[0]))
    other = simulate(bits, 4096, NOISE, query_rng(MOLECULES[1]))
    assert one == again and one != other


# -- likelihood ----------------------------------------------------------------
def test_likelihood_matches_the_bernoulli_formula():
    probs = {0: 0.9, 1: 0.2}
    ll = log_likelihood([0, 3], probs, n_bits=4)
    expected = math.log(0.9) + math.log(0.8) + math.log(EPS) + math.log(1 - EPS)
    assert ll == pytest.approx(expected)


def test_likelihood_prefers_the_candidate_the_prediction_describes():
    probs = {1: 0.95, 2: 0.9, 3: 0.1}
    assert log_likelihood([1, 2], probs, 64) > log_likelihood([1, 3], probs, 64)
    assert log_likelihood([1, 2], probs, 64) > log_likelihood([1, 2, 50], probs, 64)


def test_likelihood_orders_in_metrics():
    def result(name, ll, hit):
        return OracleResult(name, True, hit, hit, 1.0, name, log_likelihood=ll)

    outcome = QueryOutcome(results=[result("A", -9.0, False), result("B", -1.0, True)])
    assert recovery_at_k([outcome], 1) == 0.0
    assert recovery_at_k([outcome], 1, order="likelihood") == 1.0
    assert recovery_at_k([outcome], 1, order="consensus_likelihood") == 1.0


# -- model and data ------------------------------------------------------------
def test_counts_config_is_unchanged_and_old_checkpoints_still_load():
    old = {k: v for k, v in ModelConfig().to_dict().items() if not k.startswith(("fp_in", "prob"))}
    cfg = ModelConfig.from_dict(old)
    assert cfg.fp_input == "counts" and cfg.n_levels == 9


def test_probability_model_has_a_level_table(model):
    assert model.fp_encoder.count_embedding.num_embeddings == 17
    assert model.fp_encoder.bit_embedding.num_embeddings == 4096


def test_probability_mode_needs_folded_binary_fingerprints(tokenizer):
    with pytest.raises(ValueError, match="folded binary"):
        build_model(PROB, FPConfig(n_bits=4096, counts=True), tokenizer)


def test_bad_fp_input_is_refused():
    with pytest.raises(ValueError, match="fp_input"):
        ModelConfig(fp_input="probs")


def test_dataset_feeds_levels_and_varies_noise_by_epoch(tokenizer):
    ds = FingerprintDataset(
        MOLECULES, FP4096, tokenizer, prob_input=PROB.prob_input(), noise=NOISE, seed=0
    )
    first = ds[0]
    assert first["fp_counts"].min() >= 1 and first["fp_counts"].max() <= 16
    assert len(first["fp_indices"]) <= PROB.prob_max_bits
    assert torch.equal(ds[0]["fp_indices"], first["fp_indices"])  # reproducible
    ds.set_epoch(1)
    second = ds[0]
    same_bits = torch.equal(second["fp_indices"], first["fp_indices"])
    assert not (same_bits and torch.equal(second["fp_counts"], first["fp_counts"]))

    batch = collate([ds[i] for i in range(len(MOLECULES))], pad_id=tokenizer.pad_id)
    logits = build_model(PROB, FP4096, tokenizer)(
        batch["fp_indices"], batch["fp_counts"], batch["fp_mask"], batch["tokens"]
    )
    assert torch.isfinite(logits).all()


def test_dataset_without_noise_feeds_the_exact_bits(tokenizer):
    ds = FingerprintDataset(MOLECULES, FP4096, tokenizer, prob_input=PROB.prob_input())
    item = ds[0]
    assert sorted(item["fp_indices"].tolist()) == sorted(compute(MOLECULES[0], FP4096))
    assert set(item["fp_counts"].tolist()) == {16}


def test_noise_needs_a_probability_model(tokenizer):
    with pytest.raises(ValueError, match="probability"):
        FingerprintDataset(MOLECULES, FP4096, tokenizer, noise=NOISE)


# -- generation and evaluation -------------------------------------------------
def test_counts_models_refuse_probability_vectors():
    with pytest.raises(ValueError, match="probabilities"):
        _as_sparse(np.array([0.0, 0.3, 1.0]), FPConfig(n_bits=3))
    assert _as_sparse(np.array([0, 2, 1]), FPConfig(n_bits=3)) == {1: 2, 2: 1}


def test_generate_takes_probabilities(model, tmp_path):
    path = tmp_path / "prob.pt"
    model.save(path)
    loaded, _ = Morg2SmilesModel.load(path)
    assert loaded.cfg.probabilistic

    m = Morg2Smiles(loaded, torch.device("cpu"))
    dense = np.zeros(4096)
    dense[list(compute(MOLECULES[0], FP4096))] = 0.8
    out = m.generate(dense, k=5, temperature=1.0)
    assert isinstance(out, list) and len(out) <= 5
    with pytest.raises(ValueError, match="verified_only"):
        m.generate(dense, k=5, verified_only=True)


def test_evaluation_queries_with_noise_and_judges_the_true_fingerprint(model):
    report = evaluate_model(
        model,
        MOLECULES[:3],
        k=4,
        fp_unseen=[True, True, True],
        noise=NOISE,
        max_oversample=2.0,
        show_progress=False,
    )
    assert report["input_noise"]["clean_fraction"] == 0.0
    unseen = report["slices"]["fp_unseen"]
    assert "recovery_at_k_likelihood" in unseen
    assert "structure_accuracy_at_k_consensus_likelihood" in unseen


def test_noise_is_refused_for_a_counts_model(tokenizer):
    counts_model = build_model(
        ModelConfig(d_model=64, n_heads=4, d_ff=128, n_encoder_layers=1, n_decoder_layers=1),
        FPConfig(),
        tokenizer,
    )
    with pytest.raises(ValueError, match="probability"):
        evaluate_model(counts_model, MOLECULES[:1], k=2, noise=NOISE, show_progress=False)


# -- training ------------------------------------------------------------------
def test_training_runs_end_to_end_on_noisy_probabilities(tmp_path):
    from morg2smiles.train import TrainConfig, train

    shard = tmp_path / "shard"
    shard.mkdir()
    (shard / "train.smi").write_text("\n".join(MOLECULES * 4) + "\n")
    (shard / "valid.smi").write_text("\n".join(MOLECULES[:3]) + "\n")

    train_cfg = TrainConfig(
        epochs=1,
        batch_size=8,
        device="cpu",
        eval_n=2,
        eval_k=2,
        valid_loss_n=3,
        warmup_steps=1,
        noise=NOISE.to_dict(),
    )
    record = train(
        shard,
        fp_cfg=FP4096,
        model_cfg=PROB,
        train_cfg=train_cfg,
        out_dir=tmp_path / "out",
        show_progress=False,
    )
    assert record["train_config"]["noise"]["clean_fraction"] == 0.0
    model, ckpt = Morg2SmilesModel.load(tmp_path / "out" / "best.pt")
    assert model.cfg.probabilistic and model.fp_cfg.n_bits == 4096
    assert ckpt["valid_report"]["input_noise"] is not None


def test_noise_on_a_counts_model_is_refused_at_training(tmp_path):
    from morg2smiles.train import TrainConfig, train

    shard = tmp_path / "shard"
    shard.mkdir()
    (shard / "train.smi").write_text("\n".join(MOLECULES) + "\n")
    (shard / "valid.smi").write_text("\n".join(MOLECULES[:2]) + "\n")
    tiny = ModelConfig(d_model=64, n_heads=4, d_ff=128, n_encoder_layers=1, n_decoder_layers=1)
    with pytest.raises(ValueError, match="fp_input"):
        train(
            shard,
            model_cfg=tiny,
            train_cfg=TrainConfig(device="cpu", noise=NOISE.to_dict()),
            out_dir=tmp_path / "out",
            show_progress=False,
        )


def test_likelihood_uses_the_candidate_as_written_not_its_canonical_form():
    # RDKit's canonical SMILES of some aromatic systems does not parse back
    # ("can't kekulize"); seen in training as
    # 'Cc1ccc2n(Cc3ccccc3)c(O)c3c(c1-2)C=CC=CC=3'. The candidate the model wrote
    # did parse, so its likelihood must come from that spelling.
    probs = {**clean_probabilities(compute("CCO", FP4096)), 7: 0.3}
    result = OracleResult("CCO", True, False, False, 0.0, canonical="not a smiles")
    scored = with_log_likelihood(result, probs, FP4096)
    assert scored.log_likelihood == pytest.approx(
        log_likelihood(compute("CCO", FP4096), probs, FP4096.n_bits)
    )


def test_likelihood_of_an_invalid_candidate_is_left_unset():
    result = OracleResult("C1CC", False, False, False, 0.0)
    assert with_log_likelihood(result, {1: 0.9}, FP4096).log_likelihood is None
