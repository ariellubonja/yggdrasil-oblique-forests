#!/usr/bin/env python3
"""Generates the distributed-GBT dynamic load-balancing figure (SVG).

Minimal schema: per-worker FindSplits wall time is measured each round; a
worker slower than 2x the median loses a feature column to the fastest
worker, which preloads it from the shared dataset cache in the background
before ownership switches.  Same palette/geometry family as
make_column_partitioning_figure.py.
"""

OUT = "distributed_gbt_load_balancing.svg"

CANVAS_W, CANVAS_H = 1000, 620

# (name, dark, fill, hatch id) - same workers as the partitioning figure.
WORKERS = [
    ("Worker 1", "#2E5E9C", "#4A7CBB", "lbV"),
    ("Worker 2", "#B84E1F", "#D96B35", "lbD"),
    ("Worker W", "#1C7A46", "#2A8A55", "lbX"),
]
BOX_X = [60, 365, 670]
BOX_W = 270
BOX_Y, BOX_H = 110, 210
CX = [bx + BOX_W / 2 for bx in BOX_X]

TILE_W, TILE_H, TILE_GAP = 44, 92, 12
TILE_Y = BOX_Y + 48

out = []
add = out.append


def tile_x(box, slot):
    return BOX_X[box] + 25 + (TILE_W + TILE_GAP) * slot


def solid_tile(x, dark, fill, hatch):
    add(f'<rect x="{x}" y="{TILE_Y}" width="{TILE_W}" height="{TILE_H}" fill="{fill}"/>')
    add(f'<rect x="{x}" y="{TILE_Y}" width="{TILE_W}" height="{TILE_H}" fill="url(#{hatch})" '
        f'stroke="{dark}" stroke-width="1.3"/>')


def ghost_tile(x, dark, fill, label=None):
    add(f'<rect x="{x}" y="{TILE_Y}" width="{TILE_W}" height="{TILE_H}" fill="{fill}" '
        f'fill-opacity="0.16" stroke="{dark}" stroke-width="1.6" stroke-dasharray="5 4"/>')
    if label:
        add(f'<text x="{x + TILE_W / 2}" y="{TILE_Y + TILE_H / 2 + 5}" font-size="14" '
            f'font-weight="bold" fill="{dark}" text-anchor="middle">{label}</text>')


add(f'<svg xmlns="http://www.w3.org/2000/svg" width="{CANVAS_W}" height="{CANVAS_H}" '
    f'viewBox="0 0 {CANVAS_W} {CANVAS_H}" font-family="Helvetica, Arial, sans-serif">')

# ---- defs ----
add('<defs>')
for name, col in (("lbArrK", "#444444"), ("lbArrGray", "#8A8A8A")):
    add(f'<marker id="{name}" markerWidth="11" markerHeight="9" refX="8" refY="3.2" orient="auto">'
        f'<path d="M0,0 L8.5,3.2 L0,6.4 Z" fill="{col}"/></marker>')
add('<pattern id="lbV" width="7" height="7" patternUnits="userSpaceOnUse">'
    '<line x1="3.5" y1="0" x2="3.5" y2="7" stroke="#FFFFFF" stroke-opacity="0.5" stroke-width="1.7"/></pattern>')
add('<pattern id="lbD" width="8" height="8" patternUnits="userSpaceOnUse">'
    '<path d="M-2,2 L2,-2 M0,8 L8,0 M6,10 L10,6" stroke="#FFFFFF" stroke-opacity="0.5" '
    'stroke-width="1.7" fill="none"/></pattern>')
add('<pattern id="lbX" width="8" height="8" patternUnits="userSpaceOnUse">'
    '<path d="M0,0 L8,8 M8,0 L0,8" stroke="#FFFFFF" stroke-opacity="0.5" stroke-width="1.5" '
    'fill="none"/></pattern>')
add('<pattern id="lbRep" width="9" height="9" patternUnits="userSpaceOnUse">'
    '<rect width="9" height="9" fill="#F4F4F2"/>'
    '<path d="M-2,2 L2,-2 M0,9 L9,0 M7,11 L11,7" stroke="#ABABAB" stroke-width="1.3" fill="none"/></pattern>')
add('</defs>')
add(f'<rect width="{CANVAS_W}" height="{CANVAS_H}" fill="#FFFFFF"/>')

# ---- shared dataset cache (top) ----
add('<rect x="330" y="18" width="340" height="40" rx="7" fill="url(#lbRep)" '
    'stroke="#9A9A9A" stroke-width="1.3"/>')
add('<text x="500" y="43" font-size="14.5" font-weight="bold" fill="#333333" '
    'text-anchor="middle">shared dataset cache</text>')

# ---- worker boxes ----
for k, (name, dark, fill, hatch) in enumerate(WORKERS):
    bx = BOX_X[k]
    add(f'<rect x="{bx}" y="{BOX_Y}" width="{BOX_W}" height="{BOX_H}" rx="7" fill="#FDFDFC" '
        f'stroke="#C4C4C4" stroke-width="1.3"/>')
    add(f'<rect x="{bx}" y="{BOX_Y}" width="{BOX_W}" height="30" rx="7" fill="{dark}"/>')
    add(f'<rect x="{bx}" y="{BOX_Y + 20}" width="{BOX_W}" height="10" fill="{dark}"/>')
    add(f'<text x="{bx + 14}" y="{BOX_Y + 21}" font-size="15.5" font-weight="bold" '
        f'fill="#FFFFFF">{name}</text>')

_, W1_DARK, W1_FILL, W1_HATCH = WORKERS[0]
_, W2_DARK, W2_FILL, W2_HATCH = WORKERS[1]
_, WW_DARK, WW_FILL, WW_HATCH = WORKERS[2]

# Worker 1 (fast): 3 owned tiles + incoming f7 (pending, dashed).
for s in range(3):
    solid_tile(tile_x(0, s), W1_DARK, W1_FILL, W1_HATCH)
IN_X = tile_x(0, 3)
ghost_tile(IN_X, W1_DARK, W1_FILL, label="f7")
add(f'<text x="{IN_X + TILE_W / 2}" y="{TILE_Y + TILE_H + 22}" font-size="12.5" '
    f'font-weight="bold" fill="#1A1A1A" text-anchor="middle">&#9315; switch owner</text>')

# Worker 2 (slow): 2 owned tiles + f7 leaving (ghost).
for s in range(2):
    solid_tile(tile_x(1, s), W2_DARK, W2_FILL, W2_HATCH)
GO_X = tile_x(1, 2)
ghost_tile(GO_X, W2_DARK, W2_FILL, label="f7")

# Worker W: 3 owned tiles.
for s in range(3):
    solid_tile(tile_x(2, s), WW_DARK, WW_FILL, WW_HATCH)

# ---- (2) reassign arrow: Worker 2 -> Worker 1 ----
sx, sy = GO_X + TILE_W / 2, TILE_Y - 6
exx, exy = IN_X + TILE_W + 4, TILE_Y + 26
add(f'<path d="M{sx},{sy} C {sx - 60},{BOX_Y - 32} {exx + 90},{BOX_Y - 32} {exx + 4},{exy}" '
    f'fill="none" stroke="#444444" stroke-width="2.6" marker-end="url(#lbArrK)"/>')
add(f'<text x="{(sx + exx) / 2 + 14}" y="{BOX_Y - 30}" font-size="13.5" font-weight="bold" '
    f'fill="#1A1A1A" text-anchor="middle">&#9313; reassign f7</text>')

# ---- (3) preload arrow: cache -> incoming tile ----
add(f'<path d="M340,58 C 300,86 {IN_X + TILE_W / 2 - 6},110 {IN_X + TILE_W / 2},{TILE_Y - 4}" '
    f'fill="none" stroke="#8A8A8A" stroke-width="2.2" stroke-dasharray="6 4" '
    f'marker-end="url(#lbArrGray)"/>')
add('<text x="288" y="92" font-size="13.5" font-weight="bold" fill="#555555" '
    'text-anchor="end">&#9314; preload</text>')

# ---- (1) wall-time panel ----
PAN_Y, PAN_H = 358, 232
BASE_Y = PAN_Y + PAN_H - 28
add(f'<rect x="40" y="{PAN_Y}" width="920" height="{PAN_H}" rx="8" fill="#FBFBFA" '
    f'stroke="#B9B9B9" stroke-width="1.3"/>')
add(f'<text x="58" y="{PAN_Y + 26}" font-size="14" font-weight="bold" fill="#1A1A1A">'
    f'&#9312; measure FindSplits wall time</text>')

BAR_W = 56
HEIGHTS = [60, 172, 75]  # median = 75, 2x median = 150: Worker 2 crosses.
MEDIAN_H, SLOW_H = 75, 150

for label, h, yy in (("median", MEDIAN_H, BASE_Y - MEDIAN_H),
                     ("2&#215; median", SLOW_H, BASE_Y - SLOW_H)):
    add(f'<line x1="60" y1="{yy}" x2="940" y2="{yy}" stroke="#8A8A8A" stroke-width="1.2" '
        f'stroke-dasharray="5 4"/>')
    add(f'<text x="936" y="{yy - 5}" font-size="11.5" fill="#777777" text-anchor="end">{label}</text>')

for k, (name, dark, fill, hatch) in enumerate(WORKERS):
    h = HEIGHTS[k]
    x = CX[k] - BAR_W / 2
    add(f'<rect x="{x}" y="{BASE_Y - h}" width="{BAR_W}" height="{h}" fill="{fill}"/>')
    add(f'<rect x="{x}" y="{BASE_Y - h}" width="{BAR_W}" height="{h}" fill="url(#{hatch})" '
        f'stroke="{dark}" stroke-width="1.3"/>')
add(f'<line x1="90" y1="{BASE_Y}" x2="910" y2="{BASE_Y}" stroke="#444444" stroke-width="1.5"/>')
add(f'<text x="{CX[1]}" y="{BASE_Y - HEIGHTS[1] - 10}" font-size="13.5" font-weight="bold" '
    f'fill="#C0392B" text-anchor="middle">slow</text>')

# Dotted alignment guides from each worker box down to its bar.
for k in range(3):
    add(f'<line x1="{CX[k]}" y1="{BOX_Y + BOX_H + 4}" x2="{CX[k]}" y2="{BASE_Y - HEIGHTS[k] - 24}" '
        f'stroke="#CCCCCC" stroke-width="1.1" stroke-dasharray="2 4"/>')

add('</svg>')

with open(OUT, "w") as f:
    f.write("\n".join(out) + "\n")
print(f"wrote {OUT}")
