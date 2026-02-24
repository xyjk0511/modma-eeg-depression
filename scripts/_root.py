import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Allow cross-directory imports of modma_mdd_real_experiment
_modma = str(ROOT / "scripts" / "modma")
if _modma not in sys.path:
    sys.path.insert(0, _modma)
