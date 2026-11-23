"""Build docs/index.html (the project brief) and the README Highlights block
from committed results: site/card.json (alignment trace from a run of the
synthetic demo) and results/bench_render.json (render peak memory).

    python3 scripts/build_brief.py [--out docs/index.html] [--readme README.md]

Standard library only. Every number is computed here from those two files; none
is typed by hand. Components and the chart come from the vendored report-kit/.
The pipeline's own HTML report (facelapse/report_template.html) is unchanged and
is published next to this page as report.html by the Pages workflow.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "report-kit"))
import report_kit as rk  # noqa: E402
from report_kit import Raw, esc  # noqa: E402

REPO = "allenwang0/facelapse"
REPO_URL = f"https://github.com/{REPO}"
PAGE_URL = "https://allenwang0.github.io/facelapse/"
HOME_URL = "https://allenwang0.github.io/"
README_START = "<!-- results:start -->"
README_END = "<!-- results:end -->"
THIN = "\u202f"  # narrow no-break space: a number never wraps from its unit


def load(rel):
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def git_last_commit(rel):
    try:
        out = subprocess.run(
            ["git", "log", "-1", "--format=%H", "--", rel],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return out or None


def line_of(rel, needle):
    lines = (ROOT / rel).read_text(encoding="utf-8").splitlines()
    return next(i for i, line in enumerate(lines, 1) if needle in line)


def jitter(values):
    """Mean absolute frame-to-frame change, as scripts/make_card.py defines it."""
    return sum(abs(b - a) for a, b in zip(values, values[1:])) / (len(values) - 1)


def derive(card, bench):
    series = {s["name"]: [p[1] for p in s["points"]] for s in card["chart"]["series"]}
    raw, smooth = series["Raw fit"], series["Smoothed"]
    photos = int(re.search(r"(\d+) generated photos", card["data_note"]).group(1))
    years = re.search(r"(\d{4})-(\d{4})", card["data_note"]).groups()
    return {
        "raw": raw,
        "smooth": smooth,
        "frames": len(smooth),
        "photos": photos,
        "years": years,
        "j_raw": jitter(raw),
        "j_smooth": jitter(smooth),
        "raw_max": max(raw),
        "smooth_max": max(smooth),
        "peak_mib": bench["peak_rss_bytes"] / 2**20,
        "bench_frames": bench["frames"],
        "wh": f"{bench['width']}×{bench['height']}",
    }


def trace_chart(d, uid, width):
    height = 300 if width < 420 else (320 if width < 800 else 360)
    pts = lambda vals: [(i + 1, v) for i, v in enumerate(vals)]
    return rk.line_chart(
        [
            {"name": "raw fit", "points": pts(d["raw"])},
            {"name": "smoothed", "points": pts(d["smooth"]), "accent": True},
        ],
        uid=uid,
        width=width,
        height=height,
        title=f"Smoothing cuts frame-to-frame eye jitter from {d['j_raw']:.2f} to {d['j_smooth']:.2f} px",
        desc=(
            f"Eye-midpoint offset from the target position per frame, {d['frames']} frames of the synthetic demo. "
            f"Raw fit: mean frame-to-frame change {d['j_raw']:.2f} px, largest offset {d['raw_max']:.2f} px. "
            f"Smoothed: {d['j_smooth']:.2f} px, largest offset {d['smooth_max']:.2f} px."
        ),
        x_title="Frame",
        y_title="Eye offset from target (px)",
        fmt_x=lambda v: f"{v:g}",
        fmt_y=lambda v: f"{v:g}",
    )


def build_page(card, bench, shas):
    d = derive(card, bench)
    R = rk.ref

    def blob(rel, line=None):
        return f"{REPO_URL}/blob/{shas.get(rel) or 'master'}/{rel}" + (f"#L{line}" if line else "")

    smooth_line = line_of("facelapse/smoothing.py", '"""Smooth the SEQUENCE')
    detect_line = line_of("facelapse/synthetic.py", '"""')
    evidence = [
        (
            "E1",
            f"Jitter {d['j_raw']:.2f} → {d['j_smooth']:.2f} px over {d['frames']} frames",
            "site/card.json (run of the synthetic demo)",
            blob("site/card.json"),
            "Verified",
        ),
        (
            "E2",
            f"Peak memory {d['peak_mib']:,.0f} MiB rendering {d['bench_frames']:,} {d['wh']} frames",
            "results/bench_render.json",
            blob("results/bench_render.json"),
            "Verified",
        ),
        (
            "E3",
            "Smoothing averages canvas-space eye residuals over a window",
            f"facelapse/smoothing.py:{smooth_line}",
            blob("facelapse/smoothing.py", smooth_line),
            "Verified",
        ),
        (
            "E4",
            "The demo's faces are drawn and detection is stubbed",
            f"facelapse/synthetic.py:{detect_line}",
            blob("facelapse/synthetic.py", detect_line),
            "Verified",
        ),
    ]

    hero = Raw(
        '<section class="hero wrap" aria-labelledby="title"><div class="hero-grid"><div>'
        + rk.tag("p", "Computer vision · Python · Synthetic demo", class_="eyebrow")
        + rk.tag("h1", "facelapse", class_="hero-title", id="title")
        + rk.tag(
            "p",
            "Turns a messy photo archive into a face-aging timelapse: it finds one person among group "
            "shots, duplicates and lookalike relatives, then aligns one front-facing photo per week or "
            "month.",
            class_="hero-sub",
        )
        + rk.tag(
            "p",
            Raw(
                f"The timelapse shown here is the pipeline's own output on {d['photos']} generated "
                f"photos of drawn faces ({d['years'][0]}–{d['years'][1]}), with face detection stubbed. "
                f"It shows every stage running; it measures nothing about real photos.{R('E4')}"
            ),
            class_="hero-context",
        )
        + Raw(
            '<div class="actions">'
            + rk.tag("a", "Source", href=REPO_URL, class_="btn btn-primary")
            + rk.tag("a", "Pipeline report (synthetic)", href="report.html", class_="btn")
            + "</div>"
        )
        + '</div><div class="hero-figure">'
        + rk.tag("p", "Rendered timelapse, synthetic faces", class_="card-label")
        + Raw(
            '<img src="timelapse.gif" width="360" height="450" style="display:block;width:100%;max-width:300px;'
            'height:auto;margin:0 auto;border-radius:12px" alt="Synthetic timelapse generated by facelapse: '
            'drawn faces, one per month, aligned on the eyes, with a timeline bar.">'
        )
        + "</div></div></section>"
    )

    fig = rk.figure(
        "fig-jitter",
        f"Smoothing cuts frame-to-frame eye jitter from {d['j_raw']:.2f} to {d['j_smooth']:.2f}{THIN}px",
        rk.variants(
            trace_chart(d, "jt-d", 1000), trace_chart(d, "jt-t", 640), trace_chart(d, "jt-p", 320)
        ),
        rk.tag(
            "p",
            Raw(
                f"Where each frame puts the eye midpoint relative to the target, {d['frames']} frames, synthetic demo, "
                f"single run. Jitter is the mean absolute change between consecutive frames. Smoothing trades exactness "
                f"for steadiness: smoothed frames sit up to {d['smooth_max']:.2f}{THIN}px off target, raw frames up to "
                f"{d['raw_max']:.2f}{THIN}px.{R('E1')}"
            ),
        ),
        question="Does smoothing the alignment sequence make the eyes hold still between frames?",
        table=rk.data_table(
            "Eye offset per frame (px)",
            ["Frame", "Raw fit", "Smoothed"],
            [
                [str(i + 1), f"{r:.3f}", f"{s:.3f}"]
                for i, (r, s) in enumerate(zip(d["raw"], d["smooth"]))
            ],
            numeric={0, 1, 2},
        ),
        data_href=blob("site/card.json"),
    )

    body = rk.join(
        [
            Raw('<div class="prose">'),
            rk.tag(
                "p",
                Raw(
                    "Each photo's face is turned into an ArcFace embedding and kept only if it matches seed photos of the "
                    "same person from the same era, by a margin over the next-best face; close calls go to a review page. "
                    "Turned heads, strong expressions, tiny faces and near-duplicates are dropped, the best photo per period "
                    "is chosen, and the eyes are aligned on a fixed canvas."
                ),
            ),
            rk.tag(
                "p",
                Raw(
                    f"A per-photo alignment is exact but jumpy: landmark noise shifts where the eyes land from one frame "
                    f"to the next. facelapse smooths the sequence of alignments in canvas space before warping.{R('E3')}"
                ),
            ),
            Raw("</div>"),
            fig,
            Raw('<div class="prose" style="margin-top:32px">'),
            rk.tag(
                "p",
                Raw(
                    f"Rendering streams one frame at a time: on {d['bench_frames']:,} synthetic {d['wh']} frames, the render "
                    f"step's peak resident memory was {d['peak_mib']:,.0f}{THIN}MiB on {esc(bench['cpu'])}.{R('E2')}"
                ),
            ),
            rk.tag(
                "p",
                Raw(
                    "<b>Limits.</b> Faces in the demo are drawn and detection is stubbed, so nothing here measures "
                    "identification or alignment accuracy on real photos; the face models (InsightFace <code>buffalo_l</code>) "
                    "are licensed for non-commercial research only."
                ),
            ),
            Raw("</div>"),
        ]
    )
    main_sec = rk.section(
        "brief", "Smoothing the alignment sequence steadies the eyes", body, eyebrow="How it works"
    )

    repro = rk.join(
        [
            rk.code(
                'pip install -e ".[dev]"\npython scripts/build_demo.py _site   # demo archive, GIF, pipeline report\n'
                "python scripts/make_card.py              # site/card.json (needs a clean tree)\n"
                f"python scripts/bench_render.py make /tmp/bench {d['bench_frames']}\n"
                "python scripts/bench_render.py run /tmp/bench --json results/bench_render.json\n"
                "python3 scripts/build_brief.py --readme README.md",
                "Reproduce",
            ),
            rk.methodology(
                [
                    ("Machine", f"{bench['cpu']}, {bench['machine']}, Python {bench['python']}"),
                    (
                        "Load average",
                        " / ".join(f"{v:.2f}" for v in bench["load_before"])
                        + " before, "
                        + " / ".join(f"{v:.2f}" for v in bench["load_after"])
                        + " after (1 / 5 / 15 min); shared machine",
                    ),
                    ("Repetitions", "One run each; the demo is seeded and deterministic"),
                    ("Run date", bench["date"][:10]),
                ]
            ),
            Raw('<div style="margin-top:40px"></div>'),
            rk.table(
                "Evidence",
                ["ID", "Claim", "Source", "Class"],
                [
                    [
                        Raw(f'<span id="evidence-{e.lower()}">{e}</span>'),
                        claim,
                        Raw(f'<a class="src" href="{esc(h)}">{esc(src)}</a>'),
                        cls,
                    ]
                    for e, claim, src, h, cls in evidence
                ],
                wrap_cols={1, 2},
            ),
        ]
    )
    repro_sec = rk.section("evidence", "Reproduce and trace every number", repro, eyebrow="Sources")

    sha = shas.get("results/bench_render.json") or ""
    footer = rk.footer(
        Raw(
            f"Generated by <code>scripts/build_brief.py</code> from <code>site/card.json</code> and "
            f"<code>results/bench_render.json</code> · results {esc(sha[:7])} · run {esc(bench['date'][:10])}"
        ),
        Raw(f'<a href="{HOME_URL}">Allen Wang</a> · <a href="{REPO_URL}">Source</a>'),
    )
    return rk.page(
        title="facelapse · Allen Wang",
        description=(
            f"facelapse builds a face-aging timelapse from a photo archive; on its synthetic demo, smoothing "
            f"cuts eye jitter from {d['j_raw']:.2f} to {d['j_smooth']:.2f} px."
        ),
        canonical=PAGE_URL,
        header=rk.report_header(HOME_URL, "← Allen Wang", [("Source", REPO_URL)]),
        main=rk.join([hero, main_sec, repro_sec]),
        footer=footer,
    )


def readme_block(card, bench):
    d = derive(card, bench)
    return "\n".join(
        [
            README_START,
            "- Works on a real archive, not a daily-selfie habit: reads folders and Google Takeout zips in place, "
            "including iPhone HEIC photos.",
            "- Tells you apart from a sibling: your face must beat the next-best face by a margin, and close calls go "
            "to a review page instead of being guessed.",
            f"- Smooths the alignment sequence: on the synthetic demo, frame-to-frame eye jitter drops from "
            f"{d['j_raw']:.2f} px to {d['j_smooth']:.2f} px over {d['frames']} frames.",
            f"- Renders one frame at a time: {d['peak_mib']:,.0f} MiB peak memory for {d['bench_frames']:,} synthetic "
            f"{d['wh']} frames (`scripts/bench_render.py`).",
            README_END,
        ]
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", default="docs/index.html")
    parser.add_argument("--readme")
    args = parser.parse_args()
    card, bench = load("site/card.json"), load("results/bench_render.json")
    shas = {
        rel: git_last_commit(rel)
        for rel in (
            "site/card.json",
            "results/bench_render.json",
            "facelapse/smoothing.py",
            "facelapse/synthetic.py",
        )
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_page(card, bench, shas), encoding="utf-8")
    if args.readme:
        path = ROOT / args.readme
        text = path.read_text(encoding="utf-8")
        head, rest = text.split(README_START, 1)
        _, tail = rest.split(README_END, 1)
        path.write_text(head + readme_block(card, bench) + tail, encoding="utf-8")


if __name__ == "__main__":
    main()
