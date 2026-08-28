import importlib.util
from pathlib import Path


_train_path = Path(__file__).parents[1] / "scripts" / "train.py"
_spec = importlib.util.spec_from_file_location("mimic_lite_train", _train_path)
assert _spec is not None and _spec.loader is not None
_train = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_train)
_start_iteration = _train._start_iteration


def test_resume_starts_after_the_last_completed_iteration():
    assert _start_iteration(None, 2999) == 0
    assert _start_iteration("checkpoint_3000.pt", 2999) == 3000
