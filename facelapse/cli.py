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
import sys
from pathlib import Path

from .config import Config
from . import pipeline


def _cfg(args) -> Config:
    cfg = Config.load(getattr(args, "config", None))
    if getattr(args, "grain", None):
        cfg.bucket_grain = args.grain
    return cfg


def cmd_seed(args):
    from .cache import Cache
    from . import ingest, timestamps
    from .detection import FaceDetector
    cfg = _cfg(args)
    cache = Cache(cfg.cache_db)
    det = FaceDetector()
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
    pipeline.analyze(args.folder, _cfg(args), force=args.force)


def cmd_score(args):
    cfg = _cfg(args)
    au = None
    if not args.no_au:
        try:
            from .neutral import AUScorer
            au = AUScorer()
        except RuntimeError as e:
            print(f"[warn] {e}")
    pipeline.score(cfg, au_scorer=au)


def cmd_select(args):
    pipeline.select(_cfg(args))


def cmd_review(args):
    from pathlib import Path
    from .cache import Cache
    from . import ingest
    from .review import write_review
    cfg = _cfg(args)
    cache = Cache(cfg.cache_db)
    winners = cache.all_faces(only_selected=True)
    thumbs, dates = {}, {}
    for f in winners:
        thumbs[f.content_hash] = ingest.decode(Path(cache.image_path(f.content_hash)))
        dates[f.content_hash] = cache.image_ts(f.content_hash).value
    path = write_review(winners, thumbs, dates, cfg.out_dir)
    print(f"[review] {path}  ({len(winners)} winners)")
    cache.close()


def cmd_render(args):
    cfg = _cfg(args)
    rejects = set()
    if args.rejects and Path(args.rejects).exists():
        import json
        rejects = set(json.loads(Path(args.rejects).read_text()))
    result = pipeline.render_final(cfg, rejects=rejects)
    print(f"[render] {result['video']}")
    print(f"[render] {result['manifest']}")


def cmd_run(args):
    cfg = _cfg(args)
    pipeline.analyze(args.folder, cfg, force=args.force)
    au = None
    if not args.no_au:
        try:
            from .neutral import AUScorer
            au = AUScorer()
        except RuntimeError:
            pass
    pipeline.score(cfg, au_scorer=au)
    pipeline.select(cfg)
    result = pipeline.render_final(cfg)
    print(f"[done] {result['video']}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="facelapse")
    p.add_argument("--config", help="path to config JSON")
    p.add_argument("--grain", choices=["day", "week", "month"], help="override bucket grain")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("seed"); s.add_argument("action", choices=["add"]); s.add_argument("paths", nargs="+"); s.set_defaults(func=cmd_seed)
    s = sub.add_parser("analyze"); s.add_argument("folder"); s.add_argument("--force", action="store_true"); s.set_defaults(func=cmd_analyze)
    s = sub.add_parser("score"); s.add_argument("--no-au", action="store_true"); s.set_defaults(func=cmd_score)
    s = sub.add_parser("select"); s.set_defaults(func=cmd_select)
    s = sub.add_parser("review"); s.set_defaults(func=cmd_review)
    s = sub.add_parser("render"); s.add_argument("--rejects", help="JSON list of content_hashes to drop"); s.set_defaults(func=cmd_render)
    s = sub.add_parser("run"); s.add_argument("folder"); s.add_argument("--force", action="store_true"); s.add_argument("--no-au", action="store_true"); s.set_defaults(func=cmd_run)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
