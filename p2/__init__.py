"""Paper 2 code. Importing this package makes the pinned Paper 1 code (external/block-ncm) importable:
`blockncm` (models) and `error_types` (rule-based mistake detectors from block-ncm/experiments)."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BLOCKNCM = os.path.join(ROOT, "external", "block-ncm")
for p in (BLOCKNCM, os.path.join(BLOCKNCM, "experiments")):
    if p not in sys.path:
        sys.path.insert(0, p)

NCM_ROOT = os.path.expanduser(os.environ.get("NCM_ROOT", "~/ncm"))
RUNS = os.path.join(NCM_ROOT, "runs")
