#!/usr/bin/env python3
"""Generates the distributed-GBT column-partitioning figure (SVG).

Stages drawn: raw dataset cache (values in example order) -> per-column presort
-> presorted cache -> round-robin deal of columns to workers.  All feature
columns are assumed numerical, so the load balancer's cost sort is a no-op and
the deal is a plain stride-W round-robin.
"""

import random

OUT = "distributed_gbt_column_partitioning.svg"

CANVAS_W, CANVAS_H = 1000, 866

# Workers: (dark, fill, hatch-pattern id)
WORKERS = [
    ("Worker 1", "#2E5E9C", "#4A7CBB", "ovV"),
    ("Worker 2", "#B84E1F", "#D96B35", "ovD"),
    ("Worker W", "#1C7A46", "#2A8A55", "ovX"),
]
BOX_X = [60, 365, 670]
BOX_W = 270
FAN_X = [bx + BOX_W / 2 for bx in BOX_X]

# Cache-block geometry (shared by both tiers).
NUM_FEAT = 9
GLYPH_W, GLYPH_H = 66, 84
FEAT_X0, FEAT_PITCH = 62, 84
DIVIDER_X = 832
LABEL_X = 864
FEAT_CX = [FEAT_X0 + FEAT_PITCH * i + GLYPH_W / 2 for i in range(NUM_FEAT)]
LABEL_CX = LABEL_X + GLYPH_W / 2

TIER_A_Y, TIER_B_Y = 20, 246
TIER_H = 176
GLYPH_A_Y = TIER_A_Y + 44
GLYPH_B_Y = TIER_B_Y + 44

# Mini-bar value ladder drawn inside every feature glyph.
VALUES = [10, 18, 26, 34, 42, 50]
BAR_H, BAR_PITCH, BAR_X_OFF, BAR_Y_OFF = 7, 11, 8, 9
LABEL_CLASSES = [1, 0, 1, 1, 0, 1]  # same order in both tiers: y is never sorted

rng = random.Random(7)
SCRAMBLES = [rng.sample(VALUES, len(VALUES)) for _ in range(NUM_FEAT)]

out = []
add = out.append


def feature_glyph(x, y, widths, owner=None):
    """One column of the cache: frame, six values, optional owner foot bar."""
    add(f'<rect x="{x}" y="{y}" width="{GLYPH_W}" height="{GLYPH_H}" rx="4" fill="#F7F7F5"/>')
    for k, w in enumerate(widths):
        by = y + BAR_Y_OFF + BAR_PITCH * k
        add(f'<rect x="{x + BAR_X_OFF}" y="{by}" width="{w}" height="{BAR_H}" rx="1.5" fill="#8C97A3"/>')
    if owner is not None:
        add(f'<rect x="{x}" y="{y + GLYPH_H - 10}" width="{GLYPH_W}" height="8" fill="{owner}"/>')
    add(f'<rect x="{x}" y="{y}" width="{GLYPH_W}" height="{GLYPH_H}" rx="4" fill="none" '
        f'stroke="#A8A8A8" stroke-width="1.2"/>')


def label_glyph(x, y):
    """The label column: replicated, in example order, never presorted."""
    add(f'<rect x="{x}" y="{y}" width="{GLYPH_W}" height="{GLYPH_H}" rx="4" fill="url(#rep)" '
        f'stroke="#9A9A9A" stroke-width="1.3"/>')
    for k, cls in enumerate(LABEL_CLASSES):
        by = y + BAR_Y_OFF + BAR_PITCH * k
        fill = "#4A4A4A" if cls else "#D8D8D8"
        add(f'<rect x="{x + GLYPH_W / 2 - 9}" y="{by - 1.5}" width="18" height="{BAR_H + 3}" '
            f'rx="2" fill="#FFFFFF"/>')
        add(f'<rect x="{x + GLYPH_W / 2 - 6}" y="{by}" width="12" height="{BAR_H}" rx="1.5" '
            f'fill="{fill}" stroke="#7E7E7E" stroke-width="0.9"/>')


def cache_block(box_y, glyph_y, title, brace_text, sorted_cols, owners=None, sub=None):
    add(f'<rect x="30" y="{box_y}" width="940" height="{TIER_H}" rx="8" fill="#FBFBFA" '
        f'stroke="#B9B9B9" stroke-width="1.3"/>')
    add(f'<text x="52" y="{box_y + 26}" font-size="15.5" font-weight="bold" fill="#1A1A1A">{title}</text>')
    # N-rows brace on the left.
    add(f'<path d="M58,{glyph_y} L52,{glyph_y} L52,{glyph_y + GLYPH_H} L58,{glyph_y + GLYPH_H}" '
        f'fill="none" stroke="#9A9A9A" stroke-width="1.3"/>')
    bcy = glyph_y + GLYPH_H / 2
    add(f'<text x="41" y="{bcy}" font-size="11.5" fill="#777777" text-anchor="middle" '
        f'transform="rotate(-90 41 {bcy})">{brace_text}</text>')
    for i in range(NUM_FEAT):
        widths = sorted(VALUES) if sorted_cols else SCRAMBLES[i]
        feature_glyph(FEAT_X0 + FEAT_PITCH * i, glyph_y, widths,
                      owners[i] if owners else None)
    add(f'<line x1="{DIVIDER_X}" y1="{glyph_y - 4}" x2="{DIVIDER_X}" y2="{glyph_y + GLYPH_H + 4}" '
        f'stroke="#BFBFBF" stroke-width="1.2" stroke-dasharray="4 3"/>')
    label_glyph(LABEL_X, glyph_y)
    # Column names.
    ly = glyph_y + GLYPH_H + 18
    for i in range(NUM_FEAT):
        add(f'<text x="{FEAT_CX[i]}" y="{ly}" font-size="13" fill="#333333" '
            f'text-anchor="middle">f{i}</text>')
    add(f'<text x="{LABEL_CX}" y="{ly}" font-size="13" font-weight="bold" fill="#333333" '
        f'text-anchor="middle">y</text>')
    add(f'<text x="{LABEL_CX}" y="{ly + 15}" font-size="11.5" fill="#777777" '
        f'text-anchor="middle">{sub}</text>')


add(f'<svg xmlns="http://www.w3.org/2000/svg" width="{CANVAS_W}" height="{CANVAS_H}" '
    f'viewBox="0 0 {CANVAS_W} {CANVAS_H}" '
    f'font-family="Helvetica, Arial, sans-serif">')

# ---- defs ----
add('<defs>')
for name, col in (("arrB", "#2E5E9C"), ("arrO", "#B84E1F"), ("arrG", "#1C7A46"),
                  ("arrK", "#444444"), ("arrGray", "#8A8A8A")):
    add(f'<marker id="{name}" markerWidth="11" markerHeight="9" refX="8" refY="3.2" orient="auto">'
        f'<path d="M0,0 L8.5,3.2 L0,6.4 Z" fill="{col}"/></marker>')
add('<pattern id="ovV" width="7" height="7" patternUnits="userSpaceOnUse">'
    '<line x1="3.5" y1="0" x2="3.5" y2="7" stroke="#FFFFFF" stroke-opacity="0.5" stroke-width="1.7"/></pattern>')
add('<pattern id="ovD" width="8" height="8" patternUnits="userSpaceOnUse">'
    '<path d="M-2,2 L2,-2 M0,8 L8,0 M6,10 L10,6" stroke="#FFFFFF" stroke-opacity="0.5" '
    'stroke-width="1.7" fill="none"/></pattern>')
add('<pattern id="ovX" width="8" height="8" patternUnits="userSpaceOnUse">'
    '<path d="M0,0 L8,8 M8,0 L0,8" stroke="#FFFFFF" stroke-opacity="0.5" stroke-width="1.5" '
    'fill="none"/></pattern>')
add('<pattern id="rep" width="9" height="9" patternUnits="userSpaceOnUse">'
    '<rect width="9" height="9" fill="#F4F4F2"/>'
    '<path d="M-2,2 L2,-2 M0,9 L9,0 M7,11 L11,7" stroke="#ABABAB" stroke-width="1.3" fill="none"/></pattern>')
add('</defs>')
add(f'<rect width="{CANVAS_W}" height="{CANVAS_H}" fill="#FFFFFF"/>')

# ---- tier A: raw cache ----
cache_block(TIER_A_Y, GLYPH_A_Y,
            "① Dataset cache — raw/column_i/shard_j-of-k : fp32 values in example order",
            "N examples", sorted_cols=False, sub="(label)")

# ---- presort ----
for cx in FEAT_CX:
    add(f'<line x1="{cx}" y1="202" x2="{cx}" y2="238" stroke="#444444" stroke-width="1.8" '
        f'marker-end="url(#arrK)"/>')
add('<text x="826" y="211" font-size="12.5" font-weight="bold" fill="#1A1A1A">② Presort</text>')
add('<text x="826" y="226" font-size="12" fill="#555555">each feature column</text>')
add('<text x="826" y="240" font-size="12" fill="#555555">sorted by value</text>')

# ---- tier B: presorted cache, columns coloured by their new owner ----
owners = [WORKERS[i % 3][2] for i in range(NUM_FEAT)]
cache_block(TIER_B_Y, GLYPH_B_Y,
            "③ Presorted cache — indexed/column_i : each feature column in value order",
            "N values, ascending", sorted_cols=True, owners=owners, sub="(unchanged)")

# ---- deal caption ----
add('<text x="500" y="442" font-size="13.5" text-anchor="middle" fill="#333333">'
    '<tspan font-weight="bold" fill="#1A1A1A">④ Round-robin deal</tspan>'
    ' — worker w owns every W-th presorted column ⇒ ≈ F/W columns each</text>')

# ---- fan: features to their owner, label to everyone ----
FAN_Y0, FAN_Y1 = 452, 530
for i in range(NUM_FEAT):
    dark = WORKERS[i % 3][1]
    add(f'<line x1="{FEAT_CX[i]}" y1="{FAN_Y0}" x2="{FAN_X[i % 3]}" y2="{FAN_Y1}" '
        f'stroke="{dark}" stroke-width="1.8" stroke-opacity="0.9"/>')
for fx in FAN_X:
    add(f'<line x1="{LABEL_CX}" y1="{FAN_Y0}" x2="{fx}" y2="{FAN_Y1}" stroke="#8A8A8A" '
        f'stroke-width="1.6" stroke-dasharray="5 4"/>')
for k, fx in enumerate(FAN_X):
    add(f'<line x1="{fx}" y1="{FAN_Y1}" x2="{fx}" y2="564" stroke="{WORKERS[k][1]}" '
        f'stroke-width="2.6" marker-end="url(#arr{["B", "O", "G"][k]})"/>')

# ---- workers ----
WY, WH = 568, 230
for k, (name, dark, fill, hatch) in enumerate(WORKERS):
    bx = BOX_X[k]
    add(f'<rect x="{bx}" y="{WY}" width="{BOX_W}" height="{WH}" rx="7" fill="#FDFDFC" '
        f'stroke="#C4C4C4" stroke-width="1.3"/>')
    add(f'<rect x="{bx}" y="{WY}" width="{BOX_W}" height="34" rx="7" fill="{dark}"/>')
    add(f'<rect x="{bx}" y="{WY + 22}" width="{BOX_W}" height="12" fill="{dark}"/>')
    add(f'<text x="{bx + 14}" y="{WY + 23}" font-size="16" font-weight="bold" fill="#FFFFFF">{name}</text>')
    add(f'<text x="{bx + BOX_W - 14}" y="{WY + 22}" font-size="12.5" fill="#FFFFFF" '
        f'fill-opacity="0.93" text-anchor="end">3 of 9 columns + y</text>')

    cols = [f"f{k}", f"f{k + 3}", f"f{k + 6}"]
    bar_y, bar_h = WY + 64, 104
    for j, cname in enumerate(cols):
        x = bx + 20 + 56 * j
        add(f'<text x="{x + 24}" y="{WY + 56}" font-size="14" font-weight="bold" fill="#333333" '
            f'text-anchor="middle">{cname}</text>')
        add(f'<rect x="{x}" y="{bar_y}" width="48" height="{bar_h}" fill="{fill}"/>')
        add(f'<rect x="{x}" y="{bar_y}" width="48" height="{bar_h}" fill="url(#{hatch})" '
            f'stroke="{dark}" stroke-width="1.3"/>')
    add(f'<line x1="{bx + 196}" y1="{bar_y - 6}" x2="{bx + 196}" y2="{bar_y + bar_h + 6}" '
        f'stroke="#BFBFBF" stroke-width="1.2" stroke-dasharray="4 3"/>')
    add(f'<text x="{bx + 230}" y="{WY + 56}" font-size="14" font-weight="bold" fill="#333333" '
        f'text-anchor="middle">y</text>')
    add(f'<rect x="{bx + 206}" y="{bar_y}" width="48" height="{bar_h}" fill="url(#rep)" '
        f'stroke="#9A9A9A" stroke-width="1.3"/>')
    add(f'<text x="{bx + BOX_W / 2}" y="{WY + 186}" font-size="13" fill="#555555" '
        f'text-anchor="middle">all N rows of each column</text>')
    add(f'<rect x="{bx + 12}" y="{WY + 194}" width="{BOX_W - 24}" height="26" fill="url(#rep)" '
        f'stroke="#C6C6C6" stroke-width="1.1"/>')
    add(f'<text x="{bx + BOX_W / 2}" y="{WY + 211}" font-size="12.5" fill="#333333" '
        f'text-anchor="middle">grad / hess · example→node</text>')

# ---- legend ----
add('<text x="40" y="826" font-size="13" fill="#555555">'
    'solid + hatch = owned feature column, disjoint across workers &#183; '
    'gray stripe = replicated, identical on every worker</text>')
add('<text x="40" y="846" font-size="13" fill="#555555">'
    'the label column y is never presorted and never partitioned; every worker also holds the '
    'gradients / hessians and the example→node map</text>')

add('</svg>')

with open(OUT, "w") as f:
    f.write("\n".join(out) + "\n")
print(f"wrote {OUT}")
