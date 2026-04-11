from __future__ import annotations

import json

from brain_data_migration import migrate_legacy_mixed_training_data


def main():
    summary = migrate_legacy_mixed_training_data()
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
