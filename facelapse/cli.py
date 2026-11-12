"""Command-line interface. Stages are separate subcommands so the expensive
analyze pass runs once and the cheap stages iterate against the cache.

Typical flow:
    facelapse seed add photo1.jpg photo2.jpg ...   # a few confirmed-you photos across years
    facelapse analyze  ./my_photos                 # expensive, cached
    facelapse score                                # identity/pose/neutral
    facelapse select                               # dedup/bucket/pick
    facelapse review                               # eyeball the winners
    facelapse render                               # video + manifest
    facelapse run ./my_photos                      # analyze+score+select+render in one go
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from . import pipeline
from .config import Config

SINGLE_HELP = "skip the identity gate and take the largest face (required by --backend vision)"


def _cfg(args) -> Config:
    cfg = Config.load(getattr(args, "config", None))
    if getattr(args, "grain", None):
        cfg.bucket_grain = args.grain
    person = getattr(args, "person", None)
    if person:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", person):
            raise SystemExit("--person takes letters, digits, '-' and '_' only")
        cfg.person = person
        cfg.out_dir = str(Path(cfg.out_dir) / person)
    return cfg


def _backend(args):
    """(detector, mesh) for --backend. mesh None lets analyze try MediaPipe."""
    if args.backend == "synthetic":
        from .synthetic import SyntheticDetector, SyntheticMesh

        return SyntheticDetector(), SyntheticMesh()
    if args.backend == "vision":
        from .vision import VisionDetector

        helper = Path(_cfg(args).cache_db).parent / "vision_detect"
        return VisionDetector(helper), None
    from .detection import FaceDetector

    return FaceDetector(), None


def cmd_seed(args):
    from . import ingest, timestamps
    from .cache import Cache

    cfg = _cfg(args)
    cache = Cache(cfg.cache_db, cfg.person)
    det, _ = _backend(args)
    for p in args.paths:
        path = Path(p)
        rgb = ingest.decode(path)
        emb = det.embed(rgb)
        if emb is None:
            print(f"[seed] no face found in {p}, skipped")
            continue
        ts = timestamps.resolve(path)
        cache.add_seed(str(path), ts.value, emb)
        print(f"[seed] added {p}  date={ts.value}  ({ts.confidence})")
    cache.close()


def cmd_analyze(args):
    det, mesh = _backend(args)
    pipeline.analyze(args.sources, _cfg(args), detector=det, mesh=mesh, force=args.force)


def _au_scorer(args):
    if args.no_au:
        return None
    from . import neutral

    try:
        return neutral.AUScorer()
    except RuntimeError as e:
        print(f"[warn] {e}")
        return None


def cmd_score(args):
    cfg = _cfg(args)
    au = _au_scorer(args)
    if args.apply_review:
        n = pipeline.apply_review(cfg, args.apply_review)
        print(f"[review] stored {n} promotions from {args.apply_review}")
    pipeline.score(cfg, au_scorer=au, single_person=args.single_person)


def cmd_select(args):
    pipeline.select(
        _cfg(args), allow_mtime_dates=args.allow_mtime_dates, max_frames=args.max_frames
    )


def cmd_review(args):
    from . import ingest
    from .cache import Cache
    from .review import THUMB_W, Thumb, write_review

    cfg = _cfg(args)
    cache = Cache(cfg.cache_db, cfg.person)
    winners = cache.all_faces(only_selected=True)
    ambiguous = []
    for ch in cache.hashes_with_outcome("ambiguous"):
        faces = sorted(cache.faces_of(ch), key=lambda f: f.id_sim or 0.0, reverse=True)
        ambiguous.append(faces[:2])
    thumbs, dates = {}, {}
    for ch in [f.content_hash for f in winners] + [faces[0].content_hash for faces in ambiguous]:
        rgb, scale = ingest.thumbnail(Path(cache.image_path(ch)), THUMB_W)
        thumbs[ch] = Thumb(rgb, scale)
        dates[ch] = cache.image_ts(ch).value
    path = write_review(winners, ambiguous, thumbs, dates, cfg.out_dir)
    print(f"[review] {path}  ({len(winners)} winners, {len(ambiguous)} ambiguous)")
    cache.close()


def cmd_render(args):
    cfg = _cfg(args)
    from .render import OutputOptions

    rejects = _load_rejects(args.rejects)
    opts = OutputOptions(layout=args.layout, gif=args.gif, seconds=args.seconds)
    result = pipeline.render_final(cfg, rejects=rejects, options=opts)
    print(f"[render] {result['video']}")
    print(f"[render] {result['manifest']}")
    if "gif" in result:
        print(f"[render] {result['gif']}")


def _load_rejects(path: str | None) -> set[str]:
    """Accepts the review.json download ({"rejects": [...], ...}) or a plain
    JSON list of content hashes."""
    if not path:
        return set()
    data = json.loads(Path(path).read_text())
    return set(data["rejects"] if isinstance(data, dict) else data)


def cmd_run(args):
    cfg = _cfg(args)
    det, mesh = _backend(args)
    pipeline.analyze(args.sources, cfg, detector=det, mesh=mesh, force=args.force)
    pipeline.score(cfg, au_scorer=_au_scorer(args), single_person=args.single_person)
    pipeline.select(cfg, allow_mtime_dates=args.allow_mtime_dates)
    result = pipeline.render_final(cfg, rejects=_load_rejects(args.rejects))
    print(f"[done] {result['video']}")


def cmd_calibrate(args):
    import csv

    from .cache import Cache
    from .identity import calibrate

    cfg = _cfg(args)
    cache = Cache(cfg.cache_db, cfg.person)
    triples = []
    with open(args.labels, newline="") as fh:
        for row in csv.DictReader(fh):
            ch = row.get("content_hash") or cache.hash_for_path(row.get("path", ""))
            faces = sorted(
                (f for f in cache.faces_of(ch or "") if f.id_sim is not None),
                key=lambda f: f.id_sim,
                reverse=True,
            )
            if len(faces) < 2:
                continue
            me = row.get("me_face_id", "").strip()
            top_is_me = me != "" and int(me) == faces[0].face_id
            triples.append((faces[0].id_sim, faces[1].id_sim, top_is_me))
    cache.close()
    if not triples:
        print("[calibrate] no labeled image has two scored faces; run facelapse score first")
        return 1
    out = calibrate(triples, target_far=args.target_far)
    print(
        f"[calibrate] {len(triples)} images: false accept rate {out['false_accept_rate']:.3f}, "
        f"true accept rate {out['true_accept_rate']:.2f}"
    )
    print(json.dumps({"id_margin": round(out["margin"], 4)}))
    return 0


def cmd_verify(args):
    from . import ingest, verify

    det, mesh = _backend(args)
    if mesh is None:
        try:
            from .landmarks import FaceMesh

            mesh = FaceMesh()
        except RuntimeError as e:
            print(f"[warn] {e}")
    rgb = ingest.decode(Path(args.photo))
    face, checks = verify.run_checks(det, rgb)
    dense = None
    if face is not None and mesh is not None:
        x1, y1 = (max(0, int(v)) for v in face.bbox[:2])
        dense = mesh.landmarks(pipeline._crop(rgb, face.bbox))
        if dense is not None:
            dense = dense + [x1, y1]
    for c in checks:
        status = {True: "pass", False: "FAIL", None: "inconclusive"}[c.passed]
        print(f"[verify] {c.name:<15} {status:<12} {c.detail}")
    if face is not None:
        verify.write_png(rgb, face, dense, Path(args.out))
        print(f"[verify] wrote {args.out}")
    return 0 if all(c.passed is not False for c in checks) else 1


def cmd_report(args):
    from .report import write_report

    try:
        path = write_report(_cfg(args), thumbs=not args.no_thumbs, synthetic=args.synthetic)
    except FileNotFoundError as e:
        print(f"[report] error: {e}", file=sys.stderr)
        return 1
    print(f"[report] {path}")
    return 0


def main(argv=None):
    # --config and --grain are accepted before or after the subcommand.
    # SUPPRESS keeps a subparser's unset copy from overwriting a value given
    # before the subcommand.
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--config", default=argparse.SUPPRESS, help="path to config JSON")
    shared.add_argument(
        "--person",
        default=argparse.SUPPRESS,
        help="whose timelapse: separate seeds, scores and selection in the same cache",
    )
    shared.add_argument(
        "--grain",
        choices=["day", "week", "month"],
        default=argparse.SUPPRESS,
        help="override bucket grain",
    )
    p = argparse.ArgumentParser(prog="facelapse", parents=[shared])
    sub = p.add_subparsers(dest="cmd", required=True)

    def command(name, func):
        s = sub.add_parser(name, parents=[shared])
        s.set_defaults(func=func)
        return s

    def backend_option(s):
        s.add_argument(
            "--backend",
            choices=["insightface", "vision", "synthetic"],
            default="insightface",
            help="face detector: vision is macOS-only and needs score --single-person; "
            "synthetic reads the drawn test faces",
        )

    s = command("seed", cmd_seed)
    s.add_argument("action", choices=["add"])
    s.add_argument("paths", nargs="+")
    backend_option(s)
    s = command("analyze", cmd_analyze)
    s.add_argument("sources", nargs="+", help="photo folders and/or Takeout .zip files")
    s.add_argument("--force", action="store_true")
    backend_option(s)
    s = command("score", cmd_score)
    s.add_argument("--no-au", action="store_true")
    s.add_argument("--apply-review", help="review.json whose promotions to store before scoring")
    s.add_argument("--single-person", action="store_true", help=SINGLE_HELP)
    s = command("select", cmd_select)
    s.add_argument("--allow-mtime-dates", action="store_true", help="admit photos dated by mtime")
    s.add_argument("--max-frames", type=int, help="cap winners, spread evenly across years")
    command("review", cmd_review)
    s = command("report", cmd_report)
    s.add_argument("--no-thumbs", action="store_true", help="leave image thumbnails out")
    s.add_argument("--synthetic", action="store_true", help="label the page as synthetic data")
    s = command("verify", cmd_verify)
    s.add_argument("photo", help="one of your own photos with a single clear face")
    s.add_argument("--out", default="verify.png")
    backend_option(s)
    s = command("calibrate", cmd_calibrate)
    s.add_argument(
        "labels",
        help="CSV with columns path (or content_hash) and me_face_id (blank if absent)",
    )
    s.add_argument("--target-far", type=float, default=0.01, help="allowed false-accept rate")
    s = command("render", cmd_render)
    s.add_argument("--rejects", help="review.json, or a JSON list of content hashes to drop")
    s.add_argument("--layout", choices=["canvas", "square", "story"], default="canvas")
    s.add_argument("--gif", action="store_true", help="also write a looping timelapse.gif")
    s.add_argument("--seconds", type=float, help="fixed length; fps follows, 1 s end hold")
    s = command("run", cmd_run)
    s.add_argument("sources", nargs="+", help="photo folders and/or Takeout .zip files")
    s.add_argument("--force", action="store_true")
    backend_option(s)
    s.add_argument("--no-au", action="store_true")
    s.add_argument("--allow-mtime-dates", action="store_true", help="admit photos dated by mtime")
    s.add_argument("--rejects", help="review.json, or a JSON list of content hashes to drop")
    s.add_argument("--single-person", action="store_true", help=SINGLE_HELP)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
