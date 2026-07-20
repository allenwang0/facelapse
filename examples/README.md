# Example Configurations

These are starting-point configurations for different use cases. All identity thresholds (`id_t_abs`, `id_margin`) should be **recalibrated** against your own labeled face pairs using `identity.calibrate()`.

## Usage

```bash
facelapse --config examples/balanced.json run ./my_photos
```

## Presets

### conservative.json
- **Use when**: Large photo archive (1000+ photos), want only the best shots
- **Trade-off**: High precision, lower coverage
- Tighter pose limits (±15°)
- Stricter neutral gate (0.7)
- Monthly bucketing (fewer frames)

### balanced.json
- **Use when**: Medium archive (200-1000 photos), typical use case
- **Trade-off**: Balanced precision and recall
- Default settings from main Config
- Weekly bucketing

### sparse-archive.json
- **Use when**: Small archive (<200 photos), many time gaps
- **Trade-off**: Higher coverage, more false positives in review
- Looser pose limits (±25°)
- Permissive neutral gate (1.5)
- Monthly bucketing
- **Critical**: Human review step is non-negotiable with these thresholds

## Tuning Guide

After running with a preset:

1. Check funnel counts: `[funnel]` log lines show attrition at each stage
2. If too few frames survive:
   - Loosen upstream gates first (pose, neutral)
   - Check if dedup is too aggressive
3. If review shows many bad frames:
   - Tighten neutral_max_penalty
   - Increase min_interocular_px
   - Reduce max_abs_yaw_deg / max_abs_pitch_deg
4. If identity leaks (wrong person):
   - Increase id_margin
   - Check your seed photos are correct

## Calibration

**Do not skip this.** Default identity thresholds are placeholders. To derive correct values:

```python
from facelapse.identity import calibrate
from facelapse.detection import FaceDetector

det = FaceDetector()

# Collect embeddings from labeled pairs
same_pairs = []  # [(you_2015, you_2020), (you_2018, you_2023), ...]
diff_pairs = []  # [(you, sibling), (you, parent), ...]

# Extract embeddings
for img1, img2 in same_pairs:
    emb1 = det.embed(decode(img1))
    emb2 = det.embed(decode(img2))
    same_pairs_emb.append((emb1, emb2))

result = calibrate(same_pairs_emb, diff_pairs_emb)
print(result)
# Use suggested_t_abs and suggested_margin in your config
```
