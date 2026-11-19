# facelapse

facelapse turns the messy photo archive you already have into a face-aging timelapse. It finds you among group shots, screenshots, duplicates and lookalike relatives, then picks one aligned, front-facing photo per week or month.

![Timelapse rendered from the synthetic demo archive](https://allenwang0.github.io/facelapse/timelapse.gif)

The GIF and the [live report](https://allenwang0.github.io/facelapse/) come from a synthetic demo of drawn faces.

## Highlights

- Works on a real archive, not a daily-selfie habit: reads folders and Google Takeout zips in place, including iPhone HEIC photos.
- Tells you apart from a sibling: your face must beat the next-best face by a margin, and close calls go to a review page instead of being guessed.
- Renders one frame at a time: on 1,000 synthetic 1600x1200 frames, peak memory dropped from 4,466 MB to 382 MB (`scripts/bench_render.py`).
- Face detection runs once and is cached, so changing a threshold and rerunning takes seconds.

## Run it

Runs the full pipeline on a generated archive of 60 drawn faces.

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

- Runs locally, nothing uploaded; the ML extras only download model weights on first use. Those InsightFace `buffalo_l` weights are licensed for non-commercial research only.
- Some smirks and blinks get through the expression filter, which is what the review page is for.
- The synthetic demo shows each stage works; it says nothing about accuracy on real faces.
