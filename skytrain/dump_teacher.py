"""Optional teacher probability-map dumper.

In v4, sky_workflow/work/prob/*.npy ALREADY exists (2630 frames, produced by the
v3 teacher run), so SkyTraining / EXECUTION_PLAN Phase 0 converts those .npy files
directly. This script is the fallback path ONLY if the .npy cache is missing and
teacher inference must be re-run (OpenVINO Arc GPU, ~22 ms/frame).

Not wired into the default Phase 0 flow; kept for reproducibility.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="GPU")
    args = ap.parse_args()
    # The actual teacher inference lives in the v3 project (sky_only.py /
    # stages/01_label_teacher.py). Reuse that entry point rather than duplicating.
    print("v4 reuses v3 teacher output. Run:")
    print(f"  python sky_workflow/stages/01_label_teacher.py {args.input_dir}")
    print(f"    --prob {args.out_dir} --device {args.device}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
