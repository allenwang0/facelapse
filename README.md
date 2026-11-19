# facelapse

facelapse turns the messy photo archive you already have into a face-aging timelapse. It finds you among group shots, screenshots, duplicates and lookalike relatives, then picks one aligned, neutral, front-facing photo per week or month.

![Timelapse rendered from the synthetic demo archive](https://allenwang0.github.io/facelapse/timelapse.gif)

[![Run report for the synthetic demo](https://allenwang0.github.io/facelapse/report.png)](https://allenwang0.github.io/facelapse/)

Both images come from the synthetic demo (drawn faces, stubbed detection). The [live report](https://allenwang0.github.io/facelapse/) is the same page `facelapse report` writes for your own run.

## Highlights

- **Works on a real archive, not a daily-selfie habit.** Reads folders and Google Takeout zips in place, decodes HEIC, and pairs photos with Takeout sidecars even across zips.
- **Tells you apart from a sibling.** Identity is matched against seed photos from the same era, and your face must beat the next-best face by a margin. Too-close calls go to a review page instead of being guessed.
- **Streaming render:** at 1,000 frames, peak memory dropped from 4,466 MB to 382 MB, and it no longer grows with frame count.
- **Expensive work runs once.** Detection and embeddings are cached in SQLite, so changing a threshold and rerunning takes seconds.
- **Tested without a single real photo:** every test image is generated. CI covers Python 3.10 to 3.12 with a 75% coverage floor, browser checks of the report, and the macOS Vision helper.

## Quickstart

```bash
git clone https://github.com/allenwang0/facelapse
cd facelapse
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"     # core pipeline, tests and the synthetic demo
pip install -e ".[ml]"      # InsightFace, ONNX Runtime, MediaPipe, py-feat for real photos
```

Run the whole pipeline on a generated archive of 60 drawn faces (core install only):

```bash
python tests/fixtures/make_fixtures.py
facelapse seed add tests/fixtures/seeds/*.jpg --backend synthetic
facelapse run tests/fixtures/photos --backend synthetic
facelapse report --synthetic
```

This writes `out/timelapse.mp4`, `out/manifest.json` and `out/report.html`. The archive includes a case for every filter, from a lookalike twin to a re-saved copy, and the `[funnel]` output shows each one removed.

On your own photos, with the ML extras:

```bash
facelapse seed add 2016_me.jpg 2020_me.jpg 2024_me.jpg   # a few confirmed photos of you, spread over the years
facelapse analyze ./my_photos takeout-001.zip takeout-002.zip
facelapse score
facelapse select --max-frames 180
facelapse review                                          # writes out/review.html; its button downloads review.json
facelapse score --apply-review review.json                # stores your "this one is me" picks
facelapse select --max-frames 180
facelapse render --rejects review.json --gif --seconds 8
```

Useful options: `--grain day|week|month`, `--config FILE`, `render --layout story` for a 1080x1920 phone video, and `--person NAME` on each command to build one timelapse per person from a single analysis. Photos dated only by file modification time are skipped unless you pass `select --allow-mtime-dates`.

## How it works

```
ingest (folders or Takeout zips, HEIC-aware decode, content hash, dated with provenance)
  -> detect + landmarks + ArcFace embedding + pose      [expensive, cached once]
  -> identity gate (age-matched seeds, floor + margin over the runner-up)
  -> pose filter (roll corrected later; yaw and pitch filtered, never faked)
  -> neutral filter (Action Units + eye, mouth and smile geometry)
  -> face size, perceptual dedup, trusted dates only
  -> best photo per day, week or month (optionally capped per year)
  -> align (eyes + nasion similarity transform to a fixed canvas)
  -> smooth the alignment in canvas space (removes landmark jitter)
  -> photometric normalization to a median-exposure reference
  -> human review page (reject winners, resolve ambiguous photos)
  -> timeline overlay (time-scaled bar, year ticks) + H.264, GIF, manifest
```

**Identity** (`facelapse/identity.py`). ArcFace embeddings drift as a face ages, so each face is compared with the seed photos nearest in time. The most similar face in an image must clear a floor (`id_t_abs`) and beat the runner-up by a margin (`id_margin`). The margin is what keeps a sibling out: in a crowd where you clearly match best, you pass; in a two-person photo too close to call, it goes to review. Face count never enters the decision.

**Pose** (`facelapse/pose.py`, `facelapse/align.py`). Roll is removed by the alignment transform. Yaw and pitch can't be corrected without inventing hidden parts of the face, so turned faces are dropped, and selection prefers the most frontal survivor so the head doesn't appear to swivel between frames.

**Caching.** `analyze` stores faces, embeddings, landmarks and a perceptual hash in SQLite (`cache/facelapse.db`). `score`, `select`, `review` and `render` only read the cache.

**Rendering.** A first pass keeps one exposure number per frame; the second decodes, warps, normalizes and encodes one frame at a time.

| Frames (1600x1200 JPEG) | Peak RSS before | Peak RSS now | Wall time before | Wall time now |
|---|---|---|---|---|
| 100 | 541 MB | 363 MB | 3.9 s | 2.6 s |
| 300 | 1,389 MB | 383 MB | 11.4 s | 8.8 s |
| 1,000 | 4,466 MB | 382 MB | 30.5 s | 25.3 s |

Measured with `scripts/bench_render.py` on an Apple M5 Pro against the earlier in-memory renderer; the 100-frame output is identical frame for frame.

## Tuning

`facelapse report` writes `out/report.html`, a single offline file showing the stage-by-stage funnel, a calibration explorer (similarity against margin, with threshold sliders that copy out a config), coverage over time, alignment stability before and after smoothing, and where each winner's date came from.

To set `id_margin` from your own photos, label some multi-face images in a CSV (`content_hash,me_face_id`; see [examples/README.md](examples/README.md)) and run:

```bash
facelapse calibrate labels.csv --target-far 0.01
```

It prints the smallest margin with at most a 1% false-accept rate on your labels, and the share of your photos that still pass.

## Configuration

Presets in `examples/`: `balanced.config.json` (the defaults, weekly), `conservative.config.json` (tighter gates, monthly) and `sparse-archive.config.json` (looser gates for small archives, monthly).

```bash
facelapse --config examples/conservative.config.json run ./my_photos
```

| Field | Default | Meaning |
|---|---|---|
| `id_t_abs` | 0.34 | minimum cosine similarity of the top face to the age-matched seeds |
| `id_margin` | 0.10 | required gap between the top face and the runner-up; set it with `calibrate` |
| `max_abs_yaw_deg`, `max_abs_pitch_deg` | 20 | turned faces beyond these are dropped |
| `neutral_max_penalty` | 1.0 | expression penalty above which a face is dropped |
| `min_interocular_px` | 40 | faces with eyes closer than this in the source photo are dropped |
| `bucket_grain` | week | one winner per day, week or month |
| `phash_hamming_dup_below` | 6 | perceptual-hash distance that counts as the same picture |
| `smooth_window` | 3 | frames in the alignment smoothing window; 1 turns it off |
| `photometric_mode` | hist_match | `none`, `exposure` or `hist_match` |
| `fps` | 6 | frames per second unless `render --seconds` sets the length |

All fields live in [`facelapse/config.py`](facelapse/config.py). Values without a cited source are tuned defaults, not measured constants.

## macOS without the ML stack

`--backend vision` uses Apple Vision for landmarks and pose through a small Objective-C helper compiled on first use (needs Xcode Command Line Tools). Vision has no identity embedding, so use it with `score --single-person` on a folder that contains only you.

```bash
facelapse analyze ./only_me --backend vision
facelapse score --single-person
facelapse select && facelapse render
```

## Development

```bash
pytest -q --ignore=tests/e2e && ruff check . && mypy
pip install -e ".[e2e]" && playwright install chromium && pytest -q tests/e2e
```

## Limitations and privacy

- Everything runs locally. The code makes no network requests apart from the ML extras downloading model weights on first use, and the report is a self-contained file, safe to open on private photos.
- Library assumptions marked `# VERIFY` in the code (InsightFace keypoint and pose-axis order, MediaPipe landmark indices, py-feat columns) have not been confirmed on a real photo with the ML stack. `facelapse verify` checks them on yours.
- The neutral filter lets some smirks and blinks through, which is why the review page exists. Lower `neutral_max_penalty` if blinks get through.
- Tight gates plus dedup can starve a sparse archive. Read the funnel and loosen upstream gates first.
- The synthetic demo shows the stages run and filter correctly. It says nothing about detection or identity accuracy on real faces.

## License

MIT, see [LICENSE](LICENSE). The InsightFace `buffalo_l` weights downloaded by the ML extras are licensed for non-commercial research only.
