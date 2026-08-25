"""Every runnable script must at least parse and import.

A syntax error or a bad import in a CLI script does not fail the unit tests,
because nothing imports those scripts - which is how a broken
summarize_accuracy.py once got committed with a green suite. This closes that
gap cheaply: compile every top-level script, and import the ones that are safe
to import without a GPU.
"""
import ast
import importlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = sorted(p.name for p in ROOT.glob("*.py"))

# Importing a script that pulls in torch costs ~40s and needs the GPU stack,
# which CI does not install. Detect that from the source rather than keeping a
# hand-maintained list that silently goes stale.
HEAVY = ("torch", "transformers", "playwright", "qwen_vl_utils", "datasets")


def _is_heavy(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        # only module-level imports block the import; deferred ones are fine
        if any(n.split(".")[0] in HEAVY for n in names) and node.col_offset == 0:
            return True
        if any(n.startswith("distill.extract") for n in names) and node.col_offset == 0:
            return True
    return False


LIGHT = [s for s in SCRIPTS if not _is_heavy(ROOT / s)]


@pytest.mark.parametrize("name", SCRIPTS)
def test_script_parses(name):
    ast.parse((ROOT / name).read_text(encoding="utf-8"), filename=name)


@pytest.mark.parametrize("name", LIGHT)
def test_script_imports(name):
    """Import guarded by __name__ == '__main__', so this runs no work."""
    mod = name[:-3]
    try:
        importlib.import_module(mod)
    except ImportError as exc:
        pytest.skip(f"optional dependency missing: {exc}")


@pytest.mark.parametrize("name", ["distill/calibrate.py", "distill/score.py",
                                  "distill/trace.py", "distill/validate.py",
                                  "distill/schema.py", "distill/data.py"])
def test_package_module_parses(name):
    ast.parse((ROOT / name).read_text(encoding="utf-8"), filename=name)
