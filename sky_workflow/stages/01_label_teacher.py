# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Stage 1 -- teacher labeling: frames -> sky probability + binary mask.

    python sky_workflow/stages/01_label_teacher.py datasets/images -o datasets/work/mask

Writes a mirrored tree of lossless .png masks (255 = sky, 0 = ground) and, with
--prob, a matching tree of .npy probability maps. The probability maps are what a
soft-label distillation needs later; the binary masks are what the YOLO polygon
export consumes. Generate both while the teacher is loaded -- re-running it costs
a full pass.

Interactive review (--review / --review-only)
    Shows the frame next to its mask so a human can spot teacher errors while the
    run is still fresh:

        a           previous frame   (left, as in WASD)
        d           next frame       (right)
        x           mark for deletion  -> appended to --excluded
        u           undo the deletion mark
        q / Esc     quit and save

    Note that `d` is *next*, not delete -- delete is `x`. Keeping the destructive
    action off the navigation keys matters because mis-pressing a movement key while
    reviewing hundreds of frames is easy.

    "Delete" never deletes a file. It records the frame in the exclusion list that
    stages 04 and 05 already read, so the frame is skipped downstream. That keeps
    the source frames intact and makes every decision reversible -- turn the list
    back by pressing `u`, or edit the text file by hand.

    Needs a real interactive display: the review window must actually be visible.
    On a headless box cv2.imshow does not raise, it just leaves an unmapped window
    whose waitKey then returns stray values -- so the review checks visibility and
    refuses instead of acting on input nobody typed. Every mark is printed, and the
    list is plain text, so an accidental `d` costs nothing to undo.

The teacher is JianyuanWang/skyseg trimmed to a single output. Two things about it
are easy to get wrong and both fail silently:

* It needs ImageNet normalization. Feeding plain /255 returns garbage that still
  looks like a plausible mask.
* It is fixed at 320x320, so a 640x480 frame is squashed to square before
  inference and the result is stretched back. The boundary lands in the right
  place, but the teacher *judged* a distorted image.

--device GPU runs the same graph through OpenVINO on the Arc iGPU at ~22 ms/frame
instead of ~174 ms on CPU, with masks differing by 0-2 pixels.
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sky_only import EXTENSIONS, IMG_MEAN, IMG_STD, NET_SIZE, Progress, ensure_teacher, make_session  # noqa: E402

# Sky tint for the review overlay: warm so it reads against grass and concrete.
SKY_TINT = (255, 150, 40)


def probability(session, image: np.ndarray) -> np.ndarray:
    """Sky probability in [0, 1], resampled back to the frame's own resolution."""
    height, width = image.shape[:2]
    rgb = cv2.cvtColor(cv2.resize(image, (NET_SIZE, NET_SIZE)), cv2.COLOR_BGR2RGB)
    blob = ((rgb.astype(np.float32) / 255.0 - IMG_MEAN) / IMG_STD).transpose(2, 0, 1)
    raw = session.run(None, {session.get_inputs()[0].name: np.ascontiguousarray(blob[None])})[0]
    return cv2.resize(raw[0, 0], (width, height), interpolation=cv2.INTER_LINEAR)


# ---- review branch ---------------------------------------------------------
def load_excluded(path: Path) -> set:
    """Read the exclusion list; entries may carry any suffix, only stems are compared."""
    path = Path(path)
    if not path.is_file():
        return set()
    return {Path(line.strip()).stem for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def save_excluded(path: Path, stems: set) -> None:
    """Write the exclusion list, one stem per line (suffix-agnostic)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{stem}\n" for stem in sorted(stems)), encoding="utf-8")


def overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Tint the sky region so the boundary is visible at thumbnail size."""
    vis = image.copy()
    sky = mask > 127
    tint = np.array(SKY_TINT, dtype=np.float32)
    vis[sky] = (vis[sky].astype(np.float32) * 0.5 + tint * 0.5).astype(np.uint8)
    return vis


def compose(image: np.ndarray, mask: np.ndarray | None, lines: list, pane_width: int = 640):
    """Stack original | overlay above an info strip."""
    height, width = image.shape[:2]
    pane = (pane_width, int(round(pane_width * height / width)))
    left = cv2.resize(image, pane)
    right = cv2.resize(overlay(image, mask) if mask is not None else image, pane)
    if mask is None:  # nothing to compare against yet
        cv2.putText(right, "no mask yet", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    top = np.hstack([left, right])
    strip = np.zeros((72, top.shape[1], 3), np.uint8)
    for i, line in enumerate(lines):
        cv2.putText(strip, line, (8, 22 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (230, 230, 230), 1, cv2.LINE_AA)
    return np.vstack([top, strip])


def review(files: list, root: Path, mask_dir: Path, excluded_path: Path) -> set:
    """Interactive pass over frames and their masks; returns the exclusion set."""
    excluded = load_excluded(excluded_path)
    total = len(files)
    index = 0
    window = "teacher review  (a prev / d next / x delete / u undo / q quit)"
    while 0 <= index < total:
        path = files[index]
        rel = path.relative_to(root)
        image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            print(f"  skip unreadable {rel}")
            index = min(index + 1, total - 1)
            continue

        mask_path = mask_dir / rel.with_suffix(".png")
        mask = None
        if mask_path.is_file():
            mask = cv2.imdecode(np.fromfile(str(mask_path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

        header = f"{index + 1}/{total}   {rel.as_posix()}"
        if mask is not None:
            header += f"   sky={float((mask > 127).mean()):.1%}"
        lines = [header, "a previous    d next    x delete    u undo    q quit"]
        if path.stem in excluded:
            lines.append(f"MARKED FOR EXCLUSION ({len(excluded)} total) -- stages 04 and 05 will skip it")

        try:
            cv2.imshow(window, compose(image, mask, lines))
            # On a headless box imshow may not fail but never maps a window, and
            # waitKey then returns junk -- which would silently mark frames for
            # deletion. Require a visible window before reading any key.
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                raise cv2.error("window is not visible")
            pressed = cv2.waitKey(0) & 0xFF
        except cv2.error as error:  # no X display / window closed by the user
            cv2.destroyAllWindows()
            print(f"review needs an interactive display ({error}); "
                  f"edit {excluded_path} by hand instead", file=sys.stderr)
            return excluded

        if pressed in (ord("q"), 27):  # q / Esc
            break
        if pressed == ord("d"):  # forward (right, as in WASD)
            index = min(index + 1, total - 1)
        elif pressed == ord("a"):  # back (left)
            index = max(index - 1, 0)
        elif pressed == ord("x"):  # delete -> exclusion list
            excluded.add(path.stem)
            save_excluded(excluded_path, excluded)
            print(f"  excluded {path.stem}")
        elif pressed == ord("u"):  # undo
            excluded.discard(path.stem)
            save_excluded(excluded_path, excluded)
            print(f"  restored {path.stem}")

    cv2.destroyAllWindows()
    save_excluded(excluded_path, excluded)
    print(f"review done: {len(excluded)} frame(s) marked for exclusion -> {excluded_path}")
    return excluded


def main() -> int:
    here = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path)
    parser.add_argument("-o", "--out", type=Path, default=here / "work/mask")
    parser.add_argument("--prob", type=Path, default=None, help="also write float32 .npy maps here")
    parser.add_argument("-m", "--model", type=Path, default=here / "models/skyseg_teacher.onnx")
    parser.add_argument("-t", "--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="auto", help="auto/GPU/NPU use OpenVINO; cpu uses onnxruntime")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--review", action="store_true", help="review frames vs masks after labeling")
    parser.add_argument("--review-only", action="store_true",
                        help="skip teacher inference, just review the existing masks")
    parser.add_argument("--excluded", type=Path, default=None,
                        help="exclusion list (default: sibling of --out, excluded.txt)")
    args = parser.parse_args()

    files = sorted(p for p in args.input.rglob("*") if p.suffix.lower() in EXTENSIONS)
    if args.limit:
        files = files[: args.limit]
    if not files:
        print(f"no images under {args.input}", file=sys.stderr)
        return 1

    excluded_path = args.excluded or (args.out.parent / "excluded.txt")

    if args.review_only:  # no teacher needed: masks are already on disk
        review(files, args.input, args.out, excluded_path)
        return 0

    try:
        args.model = ensure_teacher(args.model)
    except Exception as error:  # offline or blocked: say so plainly instead of a stack trace
        print(f"could not obtain teacher {args.model}: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    if not args.model.is_file():
        print(f"missing teacher {args.model}", file=sys.stderr)
        return 1

    session = make_session(args.model, args.device)
    root = args.input
    started = time.perf_counter()
    progress = Progress(len(files), label="  teacher", unit=" frames")
    sky_mean = 0.0
    count = 0
    for count, path in enumerate(files, 1):
        image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            print(f"  skip unreadable {path}")
            continue
        prob = probability(session, image)
        sky_mean += prob.mean()
        progress.update(count)
        rel = path.relative_to(root)
        mask_target = args.out / rel.with_suffix(".png")
        mask_target.parent.mkdir(parents=True, exist_ok=True)
        mask_target.write_bytes(cv2.imencode(".png", (prob >= args.threshold).astype(np.uint8) * 255)[1].tobytes())
        if args.prob:
            prob_target = args.prob / rel.with_suffix(".npy")
            prob_target.parent.mkdir(parents=True, exist_ok=True)
            np.save(prob_target, prob.astype(np.float32))
    progress.close()
    rate = len(files) / max(time.perf_counter() - started, 1e-6)
    print(f"{args.out}  ({len(files)} masks, device={getattr(session, 'device', 'onnxruntime CPU')}, "
          f"{rate:.1f} img/s, mean sky={sky_mean / max(count, 1):.1%})")

    if args.review:
        review(files, root, args.out, excluded_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
