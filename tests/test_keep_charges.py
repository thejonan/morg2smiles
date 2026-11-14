"""Tests for the optional ``uncharge`` step of standardisation, and the guards
that keep shards built with different settings apart.
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import pytest
from rdkit import Chem

from morg2smiles.data.prepare import StandardizeConfig, prepare, shard_dir, standardize

KEEP = StandardizeConfig(uncharge=False)


def canonical(smiles: str) -> str:
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))


# -- standardisation -----------------------------------------------------------
def test_default_still_uncharges():
    assert StandardizeConfig().uncharge is True
    assert standardize("CC(=O)[O-].[Na+]", StandardizeConfig()) == (canonical("CC(=O)O"), "ok")


def test_keep_charges_keeps_the_largest_fragments_charge():
    assert standardize("CC(=O)[O-].[Na+]", KEEP) == (canonical("CC(=O)[O-]"), "ok")


def test_keep_charges_still_strips_salts_and_stereo():
    out, reason = standardize("C[C@H]([NH3+])C(=O)[O-].Cl", KEEP)
    assert reason == "ok"
    assert "." not in out and "@" not in out
    assert "+" in out and "-" in out


def test_permanent_charges_survive_either_way():
    """Quaternary ammonium has no proton to remove, so uncharging leaves it."""
    for cfg in (StandardizeConfig(), KEEP):
        out, reason = standardize("C[N+](C)(C)CCO.[Cl-]", cfg)
        assert reason == "ok" and "[N+]" in out


def test_settings_are_recorded():
    assert StandardizeConfig().to_dict()["uncharge"] is True
    assert KEEP.to_dict()["uncharge"] is False


# -- shard folders -------------------------------------------------------------
def test_keep_charges_shards_get_their_own_folder():
    data = Path("data")
    assert shard_dir(data, "1m") == data / "shards" / "1m"
    assert shard_dir(data, "1m", StandardizeConfig()) == data / "shards" / "1m"
    assert shard_dir(data, "1m", KEEP) == data / "shards" / "1m-keep-charges"


def _fake_shard(data_dir: Path, standardize_settings: dict) -> Path:
    out = data_dir / "shards" / "1m"
    out.mkdir(parents=True)
    (out / "meta.json").write_text(json.dumps({"standardize": standardize_settings}))
    return out


def test_an_existing_shard_built_differently_is_refused(tmp_path):
    _fake_shard(tmp_path, StandardizeConfig().to_dict())
    with pytest.raises(click.ClickException, match="max_tokens"):
        prepare("1m", data_dir=tmp_path, cfg=StandardizeConfig(max_tokens=100))


def test_shards_from_before_the_option_count_as_uncharged(tmp_path):
    old = StandardizeConfig().to_dict()
    del old["uncharge"]
    out = _fake_shard(tmp_path, old)
    assert prepare("1m", data_dir=tmp_path) == out


# -- rebuilding ----------------------------------------------------------------
def test_overwrite_does_not_leave_stale_masks(tmp_path):
    """Cached fp_unseen masks describe the old molecules and must not survive."""
    source = tmp_path / "chemreps.txt"
    rows = [f"CHEMBL{n}\t{'C' * n}O" for n in range(3, 60)]
    source.write_text("chembl_id\tcanonical_smiles\n" + "\n".join(rows) + "\n")

    out = prepare("10k", data_dir=tmp_path, source=source, processes=1)
    stale = out / "masks" / "fp_unseen_valid_x.json"
    stale.parent.mkdir()
    stale.write_text("{}")

    rebuilt = prepare("10k", data_dir=tmp_path, source=source, processes=1, overwrite=True)
    assert rebuilt == out
    assert not stale.exists()
    assert (out / "meta.json").exists() and (out / "train.smi").exists()
