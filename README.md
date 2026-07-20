# facelapse

Reconstruct a neutral-face aging timelapse from a chaotic personal photo archive (Google Photos / iCloud export), rather than from deliberately shot daily photos. The hard part is not rendering; it is *selection* (which photo, which face, is it you, is it neutral, is it usable) and *geometric normalization* (so successive frames read as one face changing, not a flipbook).

## What it does, as a funnel

```
ingest (HEIC-aware decode, content-hash, timestamp+confidence)
  -> detect + landmark + ArcFace embed + pose        [EXPENSIVE, cached once]
  -> identity gate (temporal reference chain + margin decision tree)
  -> pose filter (roll corrected in align; yaw/pitch filtered, never faked)
  -> neutral score (Action Units + geometric EAR/MAR/smile backstops)
  -> quality score + temporal bucket + perceptual dedup
  -> per-bucket best-face selection
  -> align (eye + nasion similarity transform to a canonical frame)
  -> photometric normalize (histogram match against a median reference)
  -> temporal parameter smoothing (kill detector jitter, keep real drift)
  -> [human review: static HTML approve/reject grid]
  -> timeline overlay (real-time-scaled bar, year ticks, gliding marker)
  -> H.264 encode + manifest
```

Every filtering stage prints pass/fail counts, because attrition multiplies and silent starvation is the main failure mode.

## Two design decisions worth knowing before you touch it

**Identity is rank-with-margin, not per-face threshold, and it never counts faces.** For each image, every face is scored against the *temporally nearest* reference embeddings (ArcFace is only partially age-invariant, so a 2016 face is judged against 2016-era you). The top face must clear an absolute floor *and* beat the runner-up by a margin. The margin is what rejects lookalike relatives. A crowded photo where you dominate passes; a two-person photo where you and a sibling are a coin-flip apart is sent to review. See `identity.py`.

**Facial angle: roll is corrected, yaw/pitch are filtered, nothing is fabricated.** Roll (in-plane tilt) is removed losslessly by the eye-alignment transform. Yaw and pitch (out-of-plane turns) cannot be corrected without inventing occluded pixels, so off-angle faces are dropped, not straightened. Among survivors, selection prefers the most frontal to minimize pose *variance* across the sequence (a set that merely stays under threshold but ranges widely still reads as a swiveling head). See `pose.py`, `align.py`.

## Install

```bash
pip install -e .                          # core: decode, geometry, render
pip install -r requirements-ml.txt        # detection, identity, expression
```

Core deps run and test the whole geometry/data/render path. The ML deps (InsightFace buffalo_l, MediaPipe Face Mesh, py-feat) back detection, identity, and neutral scoring, and download weights on first run.

## Use

```bash
# 1. seed a few HAND-CONFIRMED photos of you, spread across the years
facelapse seed add 2016_confirmed.jpg 2020_confirmed.jpg 2024_confirmed.jpg

# 2. expensive inference pass (cached; re-runnable stages read the cache)
facelapse analyze ./my_photos

# 3. cheap, re-runnable while tuning thresholds in a config JSON
facelapse score
facelapse select

# 4. eyeball the per-bucket winners, toggle rejects (console prints the list)
facelapse review

# 5. render (optionally pass the rejects JSON from review)
facelapse render --rejects rejects.json

# or the whole thing at once
facelapse run ./my_photos --grain week
```

Tune everything via a config JSON: `facelapse --config my.json run ...`. See `config.py` for every knob.

## Calibrate identity thresholds to YOUR face, not a benchmark

The default `id_t_abs` and `id_margin` are conservative placeholders. Derive real values from your own labeled pairs: same-person-across-years similarities (how low a true match legitimately drops with age) versus you-vs-relatives similarities (how high a false match climbs). `identity.calibrate()` returns suggested thresholds and flags the overlap band that belongs in human review.

## Getting the "only me" folder

Google Photos: open your People cluster, check for a *second* cluster of younger-you and merge it, select all, add to an album, then Takeout that album (the sidecar JSON carries the best timestamps). iCloud: export originals from your People album on macOS (expect HEIC). Face clustering still returns group photos and occasional relatives, which is why the identity gate stays on.

## Verify-before-trusting flags

These library specifics are marked `# VERIFY` in the code because getting them wrong corrupts output silently. Confirm on one of your own faces before a full run:

- **InsightFace keypoint order** (`detection.py`): indices 0,1 must be the two eyes; alignment and interocular distance depend on it.
- **InsightFace pose axis order** (`detection.py`): confirm the value treated as yaw actually moves with left/right head rotation.
- **MediaPipe landmark indices** (`landmarks.py`): the EAR/MAR/nasion indices are the common canonical ones; plot them on a test face.
- **py-feat AU column schema** (`neutral.py`): the AU return format is version-dependent.

## Known limitations

- The neutral filter leaks smirks and mid-blink frames; the review step is a designed stage, not a bug. Good scoring shrinks review from thousands to hundreds.
- Festival/rave shots are hostile on every axis (colored light wrecks recognition and photometric consistency; motion blur craters sharpness; you're rarely neutral). They mostly get filtered by their real defects, not by a head-count rule.
- Over-tight pose/neutral gates plus dedup can starve a sparse archive. Watch the funnel counts and loosen upstream before tightening downstream.
- v1 renders honest cuts. FILM/RIFE morph interpolation is a later toggle; when added, the date label must hold on the source date while the timeline marker glides across interpolated frames.

## Tests

```bash
python -m pytest tests/test_core.py       # geometry + data layers, no ML
python tests/e2e_smoke.py                 # full pipeline, ML stages stubbed
```

The smoke test proves the orchestration, cache, margin logic, alignment, bucketing, photometric, smoothing, timeline overlay, and encode all execute and emit a real mp4 + manifest; it stubs only the ML models, which are isolated behind `detection.py` / `landmarks.py` / `neutral.py`.
