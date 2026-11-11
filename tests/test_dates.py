"""Date handling. The timeline is only as honest as its dates: a photo dated
by file mtime (screenshots, re-saves, copies) must not enter the video unless
the user opts in, and every trustworthy source must resolve to the day the
camera recorded."""

from __future__ import annotations

from datetime import datetime

from facelapse import timestamps
from facelapse.records import TsSource
from tests.helpers import write_image


def test_exif_original_is_read_from_the_exif_sub_ifd(tmp_path):
    # Cameras write DateTimeOriginal in the Exif sub-IFD (0x8769), not IFD0.
    p = tmp_path / "camera.jpg"
    write_image(p, seed=1, exif_date=datetime(2019, 7, 4, 18, 30))
    ts = timestamps.resolve(p)
    assert ts.source == TsSource.EXIF_ORIGINAL
    assert ts.value == datetime(2019, 7, 4, 18, 30)
