"""Human-review export. The neutral filter leaks smirks and mid-blink frames;
good scoring shrinks the review from thousands to a few hundred, then the
human makes the final cut.

The page has two sections: the selected per-bucket winners (click to reject)
and the ambiguous images, where the top two faces were too close to call
(click "this one is me" on the right face). A button downloads review.json
with both lists; `facelapse score --apply-review review.json` stores the
promotions and `facelapse render --rejects review.json` drops the rejects.

Dev tool, not a product surface: static HTML, no server needed.
"""

from __future__ import annotations

import base64
import html
import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .records import Face

THUMB_W = 220


@dataclass
class Thumb:
    rgb: np.ndarray  # at most THUMB_W wide
    scale: float  # thumbnail pixels per original pixel


def _data_uri(rgb: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _winner_card(f: Face, t: Thumb, date: object) -> str:
    return f"""
    <div class="card" data-hash="{f.content_hash}">
      <img src="{_data_uri(t.rgb)}" alt="winner {html.escape(str(f.bucket))}"/>
      <div class="meta">
        <b>{html.escape(str(f.bucket))}</b> · {html.escape(str(date))}<br>
        comp {_fmt(f.composite)} · neutral {_fmt(f.neutral_penalty)} ·
        yaw {f.yaw:.0f}, pitch {f.pitch:.0f}<br>
        id_sim {_fmt(f.id_sim)} · margin {_fmt(f.id_margin)}
      </div>
    </div>"""


def _ambiguous_card(faces: list[Face], t: Thumb, date: object) -> str:
    h, w = t.rgb.shape[:2]
    boxes, buttons = [], []
    for f in faces:
        x1, y1, x2, y2 = (float(v) * t.scale for v in f.bbox)
        style = (
            f"left:{100 * x1 / w:.2f}%;top:{100 * y1 / h:.2f}%;"
            f"width:{100 * (x2 - x1) / w:.2f}%;height:{100 * (y2 - y1) / h:.2f}%"
        )
        boxes.append(f'<div class="box" style="{style}"><span>face {f.face_id}</span></div>')
        buttons.append(
            f'<button class="me" data-hash="{f.content_hash}" data-face="{f.face_id}">'
            f"face {f.face_id} is me (similarity {_fmt(f.id_sim)})</button>"
        )
    return f"""
    <div class="card amb">
      <div class="frame"><img src="{_data_uri(t.rgb)}" alt="ambiguous image"/>{"".join(boxes)}</div>
      <div class="meta">{html.escape(str(date))}<br>{"<br>".join(buttons)}</div>
    </div>"""


def write_review(
    selected: list[Face],
    ambiguous: list[list[Face]],
    thumbs: dict[str, Thumb],
    dates: dict[str, object],
    out_dir: str,
) -> str:
    """ambiguous holds, per image, its top two faces by identity similarity."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    winners = [_winner_card(f, thumbs[f.content_hash], dates.get(f.content_hash)) for f in selected]
    amb = [
        _ambiguous_card(faces, thumbs[faces[0].content_hash], dates.get(faces[0].content_hash))
        for faces in ambiguous
    ]
    page = (
        _PAGE.replace(
            "{{WINNERS}}", "\n".join(winners) or "<p>No winners. Run facelapse select.</p>"
        )
        .replace("{{AMBIGUOUS}}", "\n".join(amb) or "<p>No ambiguous images.</p>")
        .replace("{{N_WIN}}", str(len(selected)))
        .replace("{{N_AMB}}", str(len(ambiguous)))
    )
    path = out / "review.html"
    path.write_text(page)
    return str(path)


def _fmt(v: float | None) -> str:
    return "-" if v is None else f"{v:.2f}"


_PAGE = """<!doctype html><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>facelapse review</title>
<style>
 body{background:#111;color:#ddd;font:13px/1.4 -apple-system,sans-serif;margin:0;padding:16px}
 h1,h2{font-size:15px;font-weight:600;color:#ccc}
 .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:12px}
 .card{background:#1b1b1b;border:1px solid #2a2a2a;border-radius:8px;overflow:hidden}
 .card img{width:100%;display:block}
 .card.rej{opacity:.3;outline:2px solid #c44}
 .frame{position:relative}
 .box{position:absolute;border:2px solid #fc3;box-sizing:border-box}
 .box span{background:#fc3;color:#000;font-weight:700;padding:0 4px}
 .meta{padding:8px;color:#aaa;font-size:11px}
 button{font:inherit;margin:4px 0;padding:4px 8px;cursor:pointer}
 button.on{background:#fc3;color:#000}
 #save{position:sticky;top:0;font-size:14px;padding:8px 14px}
</style>
<button id="save">Download review.json</button>
<h1>Selected winners ({{N_WIN}}). Click a card to toggle reject.</h1>
<div class="grid" id="winners">{{WINNERS}}</div>
<h2>Ambiguous images ({{N_AMB}}). The top two faces were too close to call.</h2>
<div class="grid">{{AMBIGUOUS}}</div>
<script>
 const rejects=new Set(), promos=new Map();
 document.querySelectorAll('#winners .card').forEach(c=>c.onclick=()=>{
   const h=c.dataset.hash; c.classList.toggle('rej');
   c.classList.contains('rej')?rejects.add(h):rejects.delete(h);
 });
 document.querySelectorAll('button.me').forEach(b=>b.onclick=()=>{
   const h=b.dataset.hash, id=Number(b.dataset.face), was=promos.get(h)===id;
   b.parentElement.querySelectorAll('button.me').forEach(x=>x.classList.remove('on'));
   if(was){promos.delete(h)}else{promos.set(h,id);b.classList.add('on')}
 });
 document.getElementById('save').onclick=()=>{
   const data={rejects:[...rejects],
     promotions:[...promos].map(([content_hash,face_id])=>({content_hash,face_id}))};
   const a=document.createElement('a');
   a.href=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));
   a.download='review.json'; a.click();
 };
</script>"""
