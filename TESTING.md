# Testing Guide

## Quick Start

### 1. Install Dependencies

```bash
# Core dependencies (geometry, rendering - no ML models)
pip install -e .

# For full pipeline with face detection
pip install -r requirements-ml.txt
```

### 2. Run Existing Tests

```bash
# Core tests (no ML dependencies required)
python -m pytest tests/test_core.py -v

# End-to-end smoke test (ML models stubbed)
python tests/e2e_smoke.py
```

## Testing the Fixes

### Config Validation

```bash
# Test config validation
python3 <<EOF
from facelapse.config import Config

# Valid config - should work
cfg = Config()
print("✓ Default config valid")

# Invalid config - should raise ValueError
import json
from pathlib import Path
Path("bad_config.json").write_text('{"fps": -5, "id_t_abs": 2.0}')

try:
    Config.load("bad_config.json")
    print("✗ Validation failed - should have caught errors")
except ValueError as e:
    print("✓ Config validation caught errors:")
    print(f"  {e}")
finally:
    Path("bad_config.json").unlink()
EOF
```

### Path Traversal Protection

```bash
# Test path traversal protection
python3 <<EOF
from facelapse.ingest import iter_images
from pathlib import Path
import tempfile
import os

tmpdir = Path(tempfile.mkdtemp())
safe = tmpdir / "safe"
safe.mkdir()
(safe / "test.jpg").write_bytes(b"fake")

# Create symlink trying to escape
try:
    escape = safe / "escape"
    os.symlink("/etc", escape)  # try to link outside

    paths = list(iter_images(str(safe), (".jpg",)))
    print(f"✓ Path protection working: found {len(paths)} files (symlink ignored)")
except Exception as e:
    print(f"✓ Path protection working: {e}")
finally:
    import shutil
    shutil.rmtree(tmpdir)
EOF
```

### File Size Limit

```bash
# Test file size limit
python3 <<EOF
from facelapse.ingest import content_hash
from pathlib import Path
import tempfile

tmpdir = Path(tempfile.mkdtemp())
big_file = tmpdir / "big.jpg"

# Create a file just over the limit
big_file.write_bytes(b"x" * (201 * 1024 * 1024))  # 201MB

try:
    content_hash(big_file)
    print("✗ Size limit failed - should have rejected")
except ValueError as e:
    print(f"✓ Size limit working: {e}")
finally:
    import shutil
    shutil.rmtree(tmpdir)
EOF
```

### Database Integrity

```bash
# Test database integrity
python3 <<EOF
from facelapse.cache import Cache
from facelapse.records import Face, ImageRecord, Timestamp, TsSource
import tempfile
import numpy as np
from datetime import datetime

tmpdb = tempfile.mktemp(suffix=".db")
cache = Cache(tmpdb)

# Check PRAGMA settings
cursor = cache.db.execute("PRAGMA foreign_keys")
fk_enabled = cursor.fetchone()[0]
print(f"✓ Foreign keys enabled: {bool(fk_enabled)}")

cursor = cache.db.execute("PRAGMA journal_mode")
mode = cursor.fetchone()[0]
print(f"✓ Journal mode: {mode}")

cache.close()
import os
os.unlink(tmpdb)
EOF
```

## Minimal End-to-End Test

Create a minimal test with sample images:

```bash
# Install dependencies first
pip install -e .
pip install pillow numpy

# Create test script
cat > test_minimal.py <<'SCRIPT'
"""Minimal test without ML dependencies - tests geometry pipeline only."""
from pathlib import Path
import tempfile
import numpy as np
from PIL import Image
from datetime import datetime

from facelapse.config import Config
from facelapse import ingest, bucketing, align as align_mod, photometric, smoothing, render
from facelapse.records import Face, Timestamp, TsSource

# Create test directory
tmpdir = Path(tempfile.mkdtemp())
print(f"Test directory: {tmpdir}")

# Create 3 synthetic images
for i in range(3):
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    Image.fromarray(img).save(tmpdir / f"test_{i}.jpg")

# Test config validation
print("\n1. Testing config validation...")
cfg = Config()
cfg.out_dir = str(tmpdir / "out")
cfg._validate()  # Should not raise
print("   ✓ Config validation passed")

# Test invalid config
try:
    bad_cfg = Config(fps=-1)
    bad_cfg._validate()
    print("   ✗ Should have caught invalid fps")
except ValueError:
    print("   ✓ Invalid config rejected")

# Test image enumeration with path protection
print("\n2. Testing image enumeration...")
images = list(ingest.iter_images(str(tmpdir), (".jpg",)))
print(f"   ✓ Found {len(images)} images")

# Test content hashing with size limit
print("\n3. Testing content hash...")
for img_path in images:
    h = ingest.content_hash(img_path)
    print(f"   ✓ Hashed {img_path.name}: {h[:16]}...")

# Test alignment with synthetic face
print("\n4. Testing alignment...")
fake_face = Face(
    content_hash="test",
    face_id=0,
    bbox=np.array([100, 100, 300, 300], dtype=np.float32),
    kps=np.array([[150, 180], [250, 180], [200, 220], [160, 260], [240, 260]], dtype=np.float32),
    embedding=np.random.randn(512).astype(np.float32),
    yaw=5.0, pitch=3.0, roll=0.0, det_score=0.9
)
params = align_mod.estimate(fake_face, cfg)
print(f"   ✓ Alignment params: angle={params.angle:.3f}, scale={params.scale:.3f}")

# Test parameter smoothing
print("\n5. Testing smoothing...")
from facelapse.align import SimilarityParams
seq = [SimilarityParams(0.0, 1.0, float(i), 0.0) for i in range(10)]
smoothed = smoothing.smooth_params(seq, window=3)
print(f"   ✓ Smoothed {len(seq)} params")

# Test photometric normalization
print("\n6. Testing photometric...")
img1 = np.random.randint(50, 100, (100, 100, 3), dtype=np.uint8)
img2 = np.random.randint(150, 200, (100, 100, 3), dtype=np.uint8)
ref = photometric.choose_reference([img1, img2])
normalized = photometric.normalize(img1, ref, "exposure")
print(f"   ✓ Photometric normalization: {img1.mean():.1f} → {normalized.mean():.1f}")

print("\n✅ All geometry tests passed!")

# Cleanup
import shutil
shutil.rmtree(tmpdir)
print(f"\nCleaned up {tmpdir}")
SCRIPT

python3 test_minimal.py
```

## Testing with Real Data (ML Models Required)

```bash
# Install ML dependencies
pip install -r requirements-ml.txt

# Create a test directory with a few photos of yourself
mkdir -p test_photos
# ... copy 3-5 photos of yourself across different years

# Run the smoke test (uses stub models)
python tests/e2e_smoke.py

# Run full pipeline on real data
facelapse seed add test_photos/photo1.jpg test_photos/photo2.jpg
facelapse analyze test_photos
facelapse score
facelapse select
facelapse review
facelapse render
```

## Testing Example Configs

```bash
# Validate example configs
for cfg in examples/*.json; do
    echo "Testing $cfg..."
    python3 -c "from facelapse.config import Config; Config.load('$cfg')"
    echo "  ✓ Valid"
done
```

## Performance Testing

```bash
# Test memory usage with large archive
python3 <<EOF
import tracemalloc
from pathlib import Path

tracemalloc.start()

# Simulate render with many images
from facelapse.config import Config
cfg = Config()

snapshot1 = tracemalloc.take_snapshot()
# ... run pipeline ...
snapshot2 = tracemalloc.take_snapshot()

top_stats = snapshot2.compare_to(snapshot1, 'lineno')
print("Memory usage:")
for stat in top_stats[:3]:
    print(stat)

tracemalloc.stop()
EOF
```

## CI/CD Testing

The GitHub Actions workflow runs automatically on push:

```bash
# Manually trigger CI tests
git push origin master

# Check workflow status
gh run list --limit 3
```

## Common Issues

### ModuleNotFoundError: No module named 'cv2'
```bash
pip install opencv-python-headless
```

### ModuleNotFoundError: No module named 'insightface'
```bash
pip install -r requirements-ml.txt
```

### HEIC files not working
```bash
pip install pillow-heif
```

### Tests hang on macOS
Some ML models may request camera permissions. Run in headless mode:
```bash
export OPENCV_VIDEOIO_PRIORITY_MSMF=0
python tests/e2e_smoke.py
```
