import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _root import ROOT

from tests.test_modma_mdd_real_experiment import test_main_writes_metrics_json
from _pytest.monkeypatch import MonkeyPatch
import tempfile
import pathlib

m = MonkeyPatch()
with tempfile.TemporaryDirectory() as d:
    try:
        test_main_writes_metrics_json(pathlib.Path(d), m)
    except Exception as e:
        with open(str(ROOT / "err.txt"), "w") as f:
            import traceback
            traceback.print_exc(file=f)
