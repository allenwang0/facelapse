"""Minimal human-review export. The neutral filter leaks smirks and mid-blink
frames; good scoring shrinks the review from thousands to a few hundred, then
the human makes the final cut. This writes a static HTML grid of the selected
per-bucket winners (with runner-up swap hints and scores) plus a rejects file
you edit by hand to veto frames before the final render.

Dev tool, not a product surface: static HTML, no server needed.
"""
from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
from PIL import Image

from .records import Face


def _thumb_data_uri(rgb: np.ndarray, max_w: int = 220) -> str:
    img = Image.fromarray(rgb)
    if img.width > max_w:
        img = img.resize((max_w, int(img.height * max_w / img.width)))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def write_review(
    selected: list[Face], thumbs: dict[str, np.ndarray], dates: dict, out_dir: str,
) -> str:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cards = []
    for f in selected:
        dt = dates.get(f.content_hash)
        uri = _thumb_data_uri(thumbs[f.content_hash])
        cards.append(f"""
        <div class="card" data-hash="{f.content_hash}">
          <img src="{uri}"/>
          <div class="meta">
            <b>{f.bucket}</b> · {dt}<br>
            comp {f.composite:.2f} · neutral {_fmt(f.neutral_penalty)} ·
            front(yaw {f.yaw:.0f}, pitch {f.pitch:.0f})<br>
            id_sim {_fmt(f.id_sim)} · margin {_fmt(f.id_margin)}
          </div>
        </div>""")
    html = _PAGE.replace("{{CARDS}}", "\n".join(cards))
    path = out / "review.html"
    path.write_text(html)
    return str(path)


def _fmt(v) -> str:
    return "-" if v is None else f"{v:.2f}"


_PAGE = """<!doctype html><meta charset="utf-8">
<title>facelapse review</title>
<style>
 body{background:#111;color:#ddd;font:13px/1.4 -apple-system,sans-serif;margin:0;padding:16px}
 h1{font-size:15px;font-weight:600;color:#aaa}
 .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:12px}
 .card{background:#1b1b1b;border:1px solid #2a2a2a;border-radius:8px;overflow:hidden}
 .card img{width:100%;display:block}
 .card.rej{opacity:.3;outline:2px solid #822}
 .meta{padding:8px;color:#9a9a9a;font-size:11px}
</style>
<h1>Selected per-bucket winners. Click a card to toggle reject. Rejected hashes print to console for the render step.</h1>
<div class="grid">{{CARDS}}</div>
<script>
 const rej=new Set();
 document.querySelectorAll('.card').forEach(c=>c.onclick=()=>{
   const h=c.dataset.hash; c.classList.toggle('rej');
   c.classList.contains('rej')?rej.add(h):rej.delete(h);
   console.log('REJECTS', JSON.stringify([...rej]));
 });
</script>"""
