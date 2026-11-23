# facelapse

facelapse turns the messy photo archive you already have into a face-aging timelapse. It finds you among group shots, screenshots, duplicates and lookalike relatives, then picks one aligned, front-facing photo per week or month.

**[Read the project brief →](https://allenwang0.github.io/facelapse/)**

![Timelapse rendered from the synthetic demo archive (drawn faces, synthetic)](https://allenwang0.github.io/facelapse/timelapse.gif)

GIF and [pipeline report](https://allenwang0.github.io/facelapse/report.html): synthetic drawn faces, stubbed detection.

## Highlights

<!-- results:start -->
- Works on a real archive, not a daily-selfie habit: reads folders and Google Takeout zips in place, including iPhone HEIC photos.
- Tells you apart from a sibling: your face must beat the next-best face by a margin, and close calls go to a review page instead of being guessed.
- Smooths the alignment sequence: on the synthetic demo, frame-to-frame eye jitter drops from 1.62 px to 0.36 px over 36 frames.
- Renders one frame at a time: 143 MiB peak memory for 1,000 synthetic 1600×1200 frames (`scripts/bench_render.py`).
<!-- results:end -->

## Run it

The full pipeline on 60 generated drawn faces:

```bash
pip install -e ".[dev]"     # add ".[ml]" for face models on real photos
python tests/fixtures/make_fixtures.py
facelapse seed add tests/fixtures/seeds/*.jpg --backend synthetic
facelapse run tests/fixtures/photos --backend synthetic   # writes out/timelapse.mp4
facelapse report --synthetic   # writes out/report.html
```

## How it works

- Finds every face and turns it into a numeric fingerprint (an ArcFace embedding).
- Keeps a face only if it matches your seed photos from the same era, since faces change with age.
- Drops turned heads, strong expressions, tiny faces and near-duplicate photos.
- Picks the best photo per day, week or month and lines up the eyes on a fixed canvas.
- Evens out exposure, then encodes an MP4 or GIF with a timeline bar.

## Limitations

- Runs locally; the ML extras download InsightFace `buffalo_l` weights, licensed for non-commercial research only.
- Some smirks and blinks pass the expression filter; the review page catches them.
- The synthetic demo shows each stage works; it says nothing about accuracy on real faces.
