"""Stage 6 -- train the stripped YOLOv8n-seg on the assembled dataset.

    python sky_workflow/stages/06_train_yolo.py --epochs 100 --imgsz 320

device='xpu' resolves to the Arc 140T on this machine (ultralytics 8.4 accepts it);
fall back to --device cpu if a kernel is missing. --imgsz 320 is the one to watch:
ultralytics takes a single square size, so a 640x480 frame becomes 320x240 and is
then padded to 320x320, which spends roughly a quarter of the FLOPs on black
bands. It costs nothing at training time and the same padding must be reproduced
on the board, or the model sees a different image than it was fitted to.

--pretrained pulls the COCO yolov8n-seg weights into the nc=1/nm=1 graph. The head
shapes do not match and ultralytics discards those layers, but the backbone and
neck transfer, and on a 2.6k-frame dataset that is worth a lot.

About the "Speed: ... postprocess per image" line, which reads far worse than the
network does: the validator runs at conf=0.001 so it can draw a full precision-recall
curve, and that hands Python NMS thousands of candidate boxes per frame. Measured on
16 frames with the same weights and max_det=16, postprocess is 187 ms/image at
conf=0.001, 8.8 ms at 0.05 and 0.4 ms at 0.25. It is a metrics-building cost, not a
deployment cost -- nothing on the board runs at conf=0.001. Use --val-conf to opt out
if you only want the loss curves, and tools/check_inference_settings.py to price the
setting the board will actually use. Do not reach for --max-det to speed validation up:
the cap costs little there (NMS runs before it) and it removes the candidates the PR
curve is drawn from.
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=HERE / "dataset/data.yaml")
    parser.add_argument("--model", type=Path, default=HERE / "models/yolov8n-sky1-seg.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="cpu", help="xpu / cpu / mps / 0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--pretrained", default="yolov8n-seg.pt", help="'none' trains from scratch")
    parser.add_argument("--max-det", type=int, default=300,
                        help="cap on kept instances during validation; lowering it truncates "
                             "the PR curve and changes reported mAP -- see the docstring")
    parser.add_argument("--val-conf", type=float, default=None,
                        help="raise to ~0.25 to make validation post-processing cheap; "
                             "it truncates the PR curve, so reported mAP stops being "
                             "comparable to the ultralytics default of conf=0.001")
    parser.add_argument("--name", default="sky_yolov8n")
    parser.add_argument("--project", type=Path, default=HERE / "runs")
    args = parser.parse_args()

    if not args.data.is_file():
        print(f"{args.data} missing -- run stages/05_build_dataset.py first", file=sys.stderr)
        return 1

    from ultralytics import YOLO

    model = YOLO(str(args.model))
    if args.pretrained and args.pretrained.lower() != "none":
        try:
            model.load(args.pretrained)
            print(f"transferred backbone from {args.pretrained}")
        except Exception as error:
            print(f"pretrained load failed ({type(error).__name__}: {str(error)[:90]}), training from scratch")

    results = model.train(
        data=str(args.data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=args.device, workers=args.workers, project=str(args.project.resolve()), name=args.name,
        # Sky is one region covering most of the frame, but the head still emits a
        # handful of instances per frame and the deployment post-processor has to
        # union them -- tools/check_inference_settings.py measures where coverage
        # starts to fall. The default here stays at ultralytics' 300 because a cap
        # also truncates the candidate list the validation PR curve is built from:
        # two 1-epoch runs, identical except max_det 300 vs 16, printed mask mAP50
        # 0.535 and 0.14. Not a controlled comparison (fresh init each time), but
        # enough to keep the cap out of the default. Speed up validation with
        # --val-conf instead, which costs no candidates.
        max_det=args.max_det, conf=args.val_conf,
        plots=True, val=True, save=True, pretrained=False, cos_lr=True, patience=30,
    )

    # Read the directory back off the trainer. Ultralytics re-roots `project` under
    # its own runs_dir setting, so reconstructing the path here would lie about
    # where the weights landed.
    save_dir = Path(getattr(getattr(model, "trainer", None), "save_dir", args.project.resolve() / args.name))
    best = save_dir / "weights" / "best.pt"
    print(f"\nbest weights -> {best}")
    print(f"pass this to stage 7:  --weights {best}")
    print("Next: stages/07_export_onnx.py; 08_verify_dataset.py already gated the labels.")
    if results is not None:
        print("Remember that mask IoU here is measured against *teacher-derived* labels. "
              "It can never exceed the teacher, and it does not prove the teacher was right.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
