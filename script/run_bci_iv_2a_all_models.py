"""Run the BCI IV-2a formal suite through the shared PhysioNet protocol.

Global subject CV, run-level adaptation, ablations and audited publication CSVs
use the same runners and collector as train_global_cross_subject_cv.py.
See docs/bci_iv_2a_formal_suite.md for protocols and commands.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.train_global_cross_subject_cv import main as run_suite

DEFAULT_CONFIG = PROJECT_ROOT / "configs/bci_iv_2a_paper_suite.yaml"


def main(argv=None):
    return run_suite(argv, default_config=DEFAULT_CONFIG)


if __name__ == "__main__":
    raise SystemExit(main())
