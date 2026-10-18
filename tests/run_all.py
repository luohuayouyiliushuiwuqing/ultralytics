"""Lightweight pytest-free test runner for the Phase 0 sanity suite.

pytest is not installed in the yolo env; this discovers and runs every
test_* function in test_skytrain_sanity.py with plain assert reporting.

Run:  python tests/run_all.py
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import test_skytrain_sanity as m  # noqa: E402

funcs = [getattr(m, n) for n in dir(m) if n.startswith("test_") and callable(getattr(m, n))]

passed = failed = 0
for f in funcs:
    try:
        f()
        print(f"PASS {f.__name__}")
        passed += 1
    except Exception as e:  # noqa: BLE001
        print(f"FAIL {f.__name__}: {e}")
        traceback.print_exc()
        failed += 1

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
