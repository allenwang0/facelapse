# Example configurations

Starting points for different archives. Every field not listed in a preset keeps its default from `facelapse/config.py`. The identity thresholds (`id_t_abs`, `id_margin`) are placeholders in all of them; calibrate them on your own photos (below).

```bash
facelapse --config examples/balanced.config.json run ./my_photos
```

| Preset | Use when | Changes from the defaults |
|---|---|---|
| `balanced.config.json` | 200 to 1,000 usable photos | none (weekly buckets) |
| `conservative.config.json` | 1,000+ photos, you only want the best shots | yaw and pitch within 15 degrees, neutral penalty 0.7, monthly buckets |
| `sparse-archive.config.json` | under 200 photos with long gaps | yaw and pitch within 25 degrees, neutral penalty 1.5, monthly buckets; expect more frames to reject on the review page |

## Tuning

1. Read the `[funnel]` lines, or run `facelapse report`, to see which stage removed the most photos.
2. If too few frames survive, loosen the upstream gates (pose, neutral) first, then check whether dedup is grouping distinct photos.
3. If the review page shows many bad frames, lower `neutral_max_penalty`, raise `min_interocular_px`, or narrow the pose limits.
4. If the wrong person gets in, raise `id_margin` and check that your seed photos are correct.

## Calibrating the identity margin

After `facelapse score`, write a CSV that labels multi-face images from your archive: the image's path (or just its file name when that is unique) and the `face_id` that is you, or blank if you are not in it. The review page labels each face box in its ambiguous section with its `face_id`. A `content_hash` column works instead of `path`.

```text
path,me_face_id
2019/IMG_2041.jpg,0
2020/IMG_0815.jpg,
```

Then:

```bash
facelapse calibrate labels.csv --target-far 0.01
```

It prints the smallest `id_margin` whose false-accept rate on your labels is at most 1%, and the share of your own images that still pass at that margin.
