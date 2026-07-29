# Quick Start Guide

## Installation

```bash
# Clone the repository
git clone https://github.com/allenwang0/facelapse.git
cd facelapse

# Install core dependencies (no ML models)
pip install -e .

# Install ML dependencies (4GB+ models downloaded on first run)
pip install -r requirements-ml.txt
```

## Basic Usage

### 1. Prepare Seed Photos

Collect 2-5 hand-confirmed photos of yourself spread across different years:

```bash
facelapse seed add my_face_2016.jpg my_face_2020.jpg my_face_2024.jpg
```

### 2. Run the Pipeline

**Option A: All-in-one (recommended for first run)**
```bash
facelapse run ./my_photo_archive --grain week
```

**Option B: Step-by-step (for tuning thresholds)**
```bash
# Expensive: face detection, embeddings (cached)
facelapse analyze ./my_photo_archive

# Cheap: identity, pose, neutral filtering (re-runnable)
facelapse score
facelapse select

# Review winners, click to reject false positives
facelapse review

# Render final video
facelapse render
```

Output: `out/timelapse.mp4` and `out/manifest.json`

### 3. Use Example Configs

```bash
# Conservative: high precision, fewer frames
facelapse --config examples/conservative.json run ./photos

# Balanced: default settings
facelapse --config examples/balanced.json run ./photos

# Sparse archive: higher recall for small collections
facelapse --config examples/sparse-archive.json run ./photos
```

## Tuning

If you get too few/many frames, tune thresholds:

1. Create a custom config:
   ```bash
   facelapse --config /dev/null run ./photos  # uses defaults
   python3 -c "from facelapse.config import Config; Config().dump('my_config.json')"
   ```

2. Edit `my_config.json` (see `examples/README.md` for guidance)

3. Re-run cheap stages:
   ```bash
   facelapse --config my_config.json score
   facelapse --config my_config.json select
   facelapse --config my_config.json review
   facelapse --config my_config.json render
   ```

### Key Thresholds

- `id_t_abs` (0.3-0.4): Lower = more faces accepted (but more false positives)
- `id_margin` (0.08-0.12): Higher = stricter "definitely you" requirement
- `max_abs_yaw_deg` (15-25): Tighter = more frontal faces only
- `neutral_max_penalty` (0.7-1.5): Lower = stricter neutral expression

**Always calibrate identity thresholds to your face** (see `examples/README.md`).

## Testing

```bash
# Verify all fixes
python3 test_fixes.py

# Run core tests (no ML)
python -m pytest tests/test_core.py -v

# Run full smoke test (ML stubbed)
python tests/e2e_smoke.py
```

See `TESTING.md` for detailed testing instructions.

## Common Issues

### "No module named 'insightface'"
```bash
pip install -r requirements-ml.txt
```

### "HEIC file but pillow-heif not installed"
```bash
pip install pillow-heif
```

### Too few frames in output
- Loosen thresholds: increase `max_abs_yaw_deg`, `neutral_max_penalty`
- Check funnel logs: which stage rejected most frames?
- Use `sparse-archive.json` config

### Wrong person in output
- Add more seed photos of yourself
- Increase `id_margin` to reject ambiguous faces
- Check review page and toggle rejects

### Memory issues on large archives
- The render fix should handle this now (streaming instead of accumulation)
- If still an issue: process in batches or reduce `canvas_w/canvas_h`

## Output Files

- `out/timelapse.mp4` - Final video with timeline overlay
- `out/manifest.json` - Metadata: which photo → which frame, scores, dates
- `out/review.html` - Interactive review page (open in browser)
- `cache/facelapse.db` - SQLite cache (delete to re-run detection)

## Next Steps

1. **Calibrate identity thresholds** using your labeled face pairs
2. **Tune composite score weights** in config if selection isn't prioritizing the right features
3. **Experiment with bucket grain**: `day` (many frames), `week` (balanced), `month` (fewer frames)
4. **Add photometric mode**: `none`, `exposure`, or `hist_match` (default)

See the main README.md for detailed architecture documentation.
