#!/usr/bin/env python3
"""Reproduce the 2x4 seasonal webcam grid (fig_seasonal_grid.png).

Top row: a kept beach (Sant Antoni de Portmany) across the four seasons at 14:00.
Bottom row: an excluded camera (the Alcudia coastal-road view) at the same hour.
Same clock hour so the contrast is seasonal, not diurnal.

Pulls the ORIGINAL (blob-free) snapshots from the public media server, crops each
camera's baked-in timestamp/watermark band, and montages. Column season headers are
added in LaTeX, not baked in.

Run: python make_seasonal_grid.py
Needs network access to the media server.
"""
import io
import re
import urllib.request
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageOps

BASE = "https://ocupacioplatges.uib.eu/media/img/originals"
OUT = Path(__file__).resolve().parents[2] / "paper/figures/fig_seasonal_grid.png"

INCLUDED = "sant-antoni-de-portmany_1"   # kept beach (fills in summer)
EXCLUDED = "alcudia-sa-marina_1"         # excluded coastal-road view (flat)
HOUR = 14                                # 2 pm — fixed across seasons
SEASON_MONTHS = {"win": [1, 2, 12], "spr": [4, 5, 3], "sum": [7, 8, 6], "aut": [10, 11, 9]}
# per-camera overlay crop (fraction of height off the top / bottom)
CROP = {INCLUDED: (0.06, 0.0), EXCLUDED: (0.10, 0.0)}
CELL = (640, 360)
GAP_Y = 6


def list_originals():
    html = urllib.request.urlopen(f"{BASE}/", timeout=120).read().decode("utf-8", "ignore")
    idx = defaultdict(list)
    for f in re.findall(r'href="([a-z0-9-]+_[0-9]+_[0-9]{14}\.jpg)"', html):
        m = re.match(r"([a-z0-9-]+_[0-9]+)_(\d{4})(\d{2})(\d{2})(\d{2})", f)
        slug, y, mo, d, h = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5))
        if h == HOUR:
            idx[(slug, mo)].append((y, d, f))
    return idx


def pick(idx, slug, season):
    for month in SEASON_MONTHS[season]:
        cand = sorted(idx.get((slug, month), []))
        if cand:
            return cand[len(cand) // 2][2]        # median day in the preferred month
    return None


def fetch_crop(fname, top, bot):
    raw = urllib.request.urlopen(f"{BASE}/{fname}", timeout=60).read()
    im = Image.open(io.BytesIO(raw)).convert("RGB")
    w, h = im.size
    return im.crop((0, int(top * h), w, int(h - bot * h)))


def main():
    idx = list_originals()
    seasons = ["win", "spr", "sum", "aut"]
    rows = [INCLUDED, EXCLUDED]
    canvas = Image.new("RGB", (4 * CELL[0], 2 * CELL[1] + GAP_Y), (255, 255, 255))
    for r, slug in enumerate(rows):
        top, bot = CROP[slug]
        for c, s in enumerate(seasons):
            fname = pick(idx, slug, s)
            if not fname:
                print(f"[warn] no {HOUR}:00 image for {slug} in {s}")
                continue
            cell = ImageOps.fit(fetch_crop(fname, top, bot), CELL, method=Image.LANCZOS)
            canvas.paste(cell, (c * CELL[0], r * (CELL[1] + GAP_Y)))
            print(f"{slug:28s} {s} -> {fname}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUT)
    print(f"-> {OUT} ({canvas.size[0]}x{canvas.size[1]})")


if __name__ == "__main__":
    main()
