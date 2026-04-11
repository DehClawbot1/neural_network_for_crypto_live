from pathlib import Path

from rl_bootstrap import bootstrap_split_rl_aliases


def test_bootstrap_split_rl_aliases_clones_legacy_shared_rl(tmp_path: Path):
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    legacy_zip = weights_dir / "ppo_polytrader.zip"
    legacy_vec = weights_dir / "ppo_polytrader_vecnormalize.pkl"
    legacy_zip.write_bytes(b"legacy-ppo")
    legacy_vec.write_bytes(b"legacy-vec")

    created = bootstrap_split_rl_aliases(
        legacy_weights_path=legacy_zip,
        entry_weights_path=weights_dir / "ppo_entry_policy.zip",
        position_weights_path=weights_dir / "ppo_position_policy.zip",
    )

    assert str(weights_dir / "ppo_entry_policy.zip") in created
    assert str(weights_dir / "ppo_position_policy.zip") in created
    assert str(weights_dir / "ppo_entry_vecnormalize.pkl") in created
    assert str(weights_dir / "ppo_position_vecnormalize.pkl") in created
    assert (weights_dir / "ppo_entry_policy.zip").read_bytes() == b"legacy-ppo"
    assert (weights_dir / "ppo_position_policy.zip").read_bytes() == b"legacy-ppo"
    assert (weights_dir / "ppo_entry_vecnormalize.pkl").read_bytes() == b"legacy-vec"
    assert (weights_dir / "ppo_position_vecnormalize.pkl").read_bytes() == b"legacy-vec"
