from __future__ import annotations

import shutil
from pathlib import Path


def bootstrap_split_rl_aliases(
    *,
    legacy_weights_path: Path,
    entry_weights_path: Path,
    position_weights_path: Path,
):
    created = []
    if not legacy_weights_path.exists():
        return created

    legacy_vecnorm = legacy_weights_path.with_name(f"{legacy_weights_path.stem}_vecnormalize.pkl")
    alias_specs = [
        (entry_weights_path, entry_weights_path.with_name("ppo_entry_vecnormalize.pkl")),
        (position_weights_path, position_weights_path.with_name("ppo_position_vecnormalize.pkl")),
    ]
    for model_path, vecnorm_path in alias_specs:
        if not model_path.exists():
            model_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(legacy_weights_path, model_path)
            created.append(str(model_path))
        if legacy_vecnorm.exists() and not vecnorm_path.exists():
            vecnorm_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(legacy_vecnorm, vecnorm_path)
            created.append(str(vecnorm_path))
    return created
