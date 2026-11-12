# facelapse

facelapse builds a face-aging timelapse from the unsorted photo archive you already have, picking one aligned, neutral, front-facing photo of you per week or month.

![Timelapse rendered from the synthetic demo archive](https://allenwang0.github.io/facelapse/timelapse.gif)

[![Run report for the synthetic demo](https://allenwang0.github.io/facelapse/report.png)](https://allenwang0.github.io/facelapse/)

Both images come from the synthetic demo: the faces are drawn shapes and detection is stubbed. The [live report](https://allenwang0.github.io/facelapse/) is the same page `facelapse report` writes for your own run.

## Why

Aging timelapses usually need a daily selfie habit kept up for years. Most people instead have a decade of phone photos: group shots, screenshots, re-saved copies, photos of relatives who look like them, and long gaps. facelapse works on that archive. The hard part is choosing which photo, which face in it, and whether it is usable; rendering is the easy part.

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

Every stage prints how many photos it kept and stores the count, with its top rejection reasons, in the cache.

Identity. Each face is compared with the seed photos nearest in time, because ArcFace embeddings drift as a face ages. In each image the most similar face must clear a floor (`id_t_abs`) and beat the second most similar face by a margin (`id_margin`). The margin is what keeps a sibling out: a crowd where you clearly match best passes, and a two-person photo where you and a lookalike are too close to call goes to the review page instead of being guessed. Face count never enters the decision. See `facelapse/identity.py`.

Pose. Roll (head tilt in the image plane) is removed by the alignment transform. Yaw and pitch cannot be corrected without inventing hidden parts of the face, so turned faces are dropped. Among the survivors, selection prefers the most frontal face, which keeps the head from appearing to swivel between frames. See `facelapse/pose.py` and `facelapse/align.py`.

Caching. `analyze` runs detection once per photo and stores faces, embeddings, landmarks and a perceptual hash in SQLite (`cache/facelapse.db`). `score`, `select`, `review` and `render` read the cache, so you can change thresholds and rerun them in seconds. Each rerun clears the previous run's results first, so a stricter setting never leaves old winners behind.

## Install

```bash
git clone https://github.com/allenwang0/facelapse
cd facelapse
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"     # core pipeline, tests and the synthetic demo
pip install -e ".[ml]"      # InsightFace, ONNX Runtime, MediaPipe, py-feat for real photos
```

The core install runs every stage except face detection, identity and expression scoring. The ML extras download model weights on first use. InsightFace's `buffalo_l` weights are licensed for non-commercial research only; check that license before using facelapse for anything else.

## Quickstart

This runs the full pipeline on a generated archive of 60 drawn faces, so it needs only the core install:

```bash
python tests/fixtures/make_fixtures.py
facelapse seed add tests/fixtures/seeds/*.jpg --backend synthetic
facelapse run tests/fixtures/photos --backend synthetic
facelapse report --synthetic
```

The run writes `out/timelapse.mp4`, `out/manifest.json` and `out/report.html`. The archive contains a case for every filter: turned heads, smiles, closed eyes, tiny faces, a lookalike twin, strangers, a re-saved copy, screenshots dated only by file time, and photos dated by a Takeout sidecar or by filename. The `[funnel]` lines show each one being removed.

On your own photos, with the ML extras installed:

```bash
facelapse seed add 2016_me.jpg 2020_me.jpg 2024_me.jpg   # a few photos you confirm are you, spread over the years
facelapse analyze ./my_photos takeout-001.zip takeout-002.zip
facelapse score
facelapse select --max-frames 180
facelapse review                                          # writes out/review.html; its button downloads review.json
facelapse score --apply-review review.json                # stores your "this one is me" picks
facelapse select --max-frames 180
facelapse render --rejects review.json --gif --seconds 8
```

`analyze` reads folders and Google Takeout zips in place, without extracting, and pairs each photo with its Takeout sidecar even when the sidecar is in another zip. `--grain day|week|month` and `--config` work before or after the subcommand. Photos dated only by file modification time are left out unless you pass `select --allow-mtime-dates`. `render --layout story` writes a 1080x1920 video for phone stories.

To build one timelapse per person from the same archive, analyze once and pass `--person NAME` to `seed`, `score`, `select`, `review`, `render` and `report`. Each person gets their own seeds, results and `out/NAME/` folder.

Getting a cleaner archive helps. In Google Photos, open your face group, merge any second group of younger you, add everything to an album and export that album with Takeout. In Apple Photos, export originals from your People album (expect HEIC).

## Tuning

`facelapse report` writes `out/report.html`, one file with no external requests, so it is safe to open on private photos. It has five views:

1. Funnel: photos kept at each stage, with the top rejection reasons on hover.
2. Calibration explorer: one point per image, top-face similarity against margin over the runner-up, split into none, ambiguous and confident regions by the current thresholds. Two sliders move the thresholds and update the counts; "Copy config" gives the JSON for the values you chose. Click a point to see its thumbnail.
3. Coverage: photos per bucket over time, with a mark on buckets that have a winner. Gaps show as gaps.
4. Alignment stability: per frame, how far the eye midpoint sits from its target before and after smoothing.
5. Date provenance: how many winners are dated by EXIF or Takeout, and how many rest on filenames or file times.

To set `id_margin` from your own photos, label some multi-face images in a CSV (`content_hash,me_face_id`, blank when you are not in the photo) and run:

```bash
facelapse calibrate labels.csv --target-far 0.01
```

It prints the smallest margin whose false-accept rate on your labels is at most 1% and the share of your own photos that still pass. `examples/README.md` explains the CSV.

Before trusting a full run with the ML stack, run `facelapse verify one_of_your_photos.jpg`. It writes `verify.png` with the detector keypoints numbered and the MediaPipe points the code reads, and checks that mirroring the face flips yaw and roll, that a 15 degree rotation moves roll by about 15, and that the eye keypoints come in the expected order.

## Configuration

Three presets live in `examples/`: `balanced.config.json` (the defaults, weekly buckets), `conservative.config.json` (tighter pose and expression gates, monthly) and `sparse-archive.config.json` (looser gates for small archives, monthly).

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

Every field is in `facelapse/config.py`. Values without a cited source are tuned defaults, not measured constants.

## Performance

`render` streams: it measures each frame's exposure in a first pass that keeps one number per frame, then decodes, warps, normalizes and encodes one frame at a time. Peak memory no longer grows with the number of frames.

| Frames (1600x1200 JPEG) | Peak RSS before | Peak RSS now | Wall time before | Wall time now |
|---|---|---|---|---|
| 100 | 541 MB | 363 MB | 3.9 s | 2.6 s |
| 300 | 1,389 MB | 383 MB | 11.4 s | 8.8 s |
| 1,000 | 4,466 MB | 382 MB | 30.5 s | 25.3 s |

Measured with `scripts/bench_render.py` and `/usr/bin/time -l` on an Apple M5 Pro with 48 GB of memory, Python 3.12. "Before" is the render as of commit 531fc2f. The 100-frame output is identical frame for frame to that render. Across two runs, wall time varied by up to 25% and peak memory by under 5%. `select` reads the perceptual hash stored by `analyze` instead of decoding every photo again.

## macOS without the ML stack

On macOS, `--backend vision` uses Apple Vision for face landmarks and head pose. A small Objective-C helper (`facelapse/vision_detect.m`) is compiled with clang the first time you use it, so the Xcode Command Line Tools must be installed. Vision gives no identity embedding, so this backend cannot tell you apart from other people: `score --single-person` (required with this backend) skips the identity gate and takes the largest face in each photo. Use it on a folder that contains only you, such as an exported People album.

```bash
facelapse analyze ./only_me --backend vision
facelapse score --single-person
facelapse select && facelapse render
```

## Limitations

- The library assumptions marked `# VERIFY` in the code (InsightFace keypoint order and pose axis order, MediaPipe landmark indices, the py-feat column format) have not been confirmed on a real photo with the ML stack. `facelapse verify` checks them on your photo; the checks themselves are tested against a synthetic detector, including copies with swapped axes and swapped eyes.
- The neutral filter leaks smirks and mid-blink frames, which is why the review page exists. A closed-eye face with nothing else wrong scores exactly the default limit of 1.0 and passes; lower `neutral_max_penalty` if blinks get through.
- Tight pose and expression gates plus dedup can starve a sparse archive. Read the funnel and loosen the upstream gates first.
- `render --gif` keeps the 360 px GIF frames in memory; the video itself streams.
- The synthetic demo shows that the stages run and filter what they should. It says nothing about detection or identity accuracy on real faces.

## Development

```bash
pip install -e ".[dev]"
pytest -q --ignore=tests/e2e        # unit, regression and end-to-end tests on generated images
ruff check . && ruff format --check . && mypy
pip install -e ".[e2e]" && playwright install chromium
pytest -q tests/e2e                 # loads the report in a browser at 390 and 1280 px, light and dark
python scripts/make_card.py         # writes site/card.json for the portfolio site from a demo run
```

If the bundled headless Chromium cannot take screenshots on your machine, set `PLAYWRIGHT_CHANNEL=chrome` to use an installed Chrome. Tests never use real photos: `tests/fixtures/make_fixtures.py` draws every image. CI runs lint and types, the test matrix on Python 3.10 to 3.12 with a 75% coverage floor, the browser checks, and the Vision helper on macOS. `.github/workflows/pages.yml` publishes the synthetic demo report.

## License

MIT, see `LICENSE`. The InsightFace model weights the ML extras download have their own non-commercial research license.
