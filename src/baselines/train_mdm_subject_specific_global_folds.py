"""Target-only MDM run-LOO grouped by the original global test-subject folds.

Classical MDM has no neural fine-tuning checkpoint; centroids are fitted using
only the target's non-test runs. Global class centroids are not reused.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.standalone_subject_specific import main

if __name__ == "__main__":
    raise SystemExit(main("mdm"))
