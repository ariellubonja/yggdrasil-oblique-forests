#!/usr/bin/env python3
"""Generates the QuickScorer-vs-sparse-oblique schema (SVG).

Two tiers.  Tier A: axis-aligned conditions can be compiled feature-major --
every node testing feature f drops its threshold into f's bucket, thresholds
are sorted once offline, and scoring is one ascending scan per feature plus
bitmask ANDs.  Tier B: a sparse-oblique condition sum(w_i * x_fi) >= t spans
several features and compares a node-private dot product, so it has no home
bucket and no shared sorted axis; QuickScorer's engine whitelist rejects it
(register_engines.cc, AllConditionsCompatibleQuickScorerExtendedModels) and
the model is served by the generic traversal engine instead
(decision_forest_serving.cc, kNumericalObliqueProjectionIsHigher).
"""

OUT = "quickscorer_sparse_oblique.svg"

W, H = 1000, 700

BLUE_D, BLUE_F = "#2E5E9C", "#E8EFF8"
ORANGE_D, ORANGE_F = "#B84E1F", "#FBEFE7"
GREEN_D = "#1C7A46"
GRAY = "#8C97A3"
INK = "#2B2F33"
RED = "#B3261E"

FONT = 'font-family="Helvetica,Arial,sans-serif"'

out = []
add = out.append


def box(x, y, w, h, stroke, fill, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="1.6"{d}/>')


def text(x, y, s, size=14, color=INK, anchor="start", weight="normal",
         style="normal"):
    add(f'<text x="{x}" y="{y}" {FONT} font-size="{size}" fill="{color}" '
        f'text-anchor="{anchor}" font-weight="{weight}" '
        f'font-style="{style}">{s}</text>')


def arrow(x1, y1, x2, y2, color=GRAY, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    add(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
        f'stroke-width="1.8" marker-end="url(#arr)"{d}/>')


add(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
    f'viewBox="0 0 {W} {H}">')
add('<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" '
    'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
    f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{GRAY}"/></marker></defs>')
add(f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>')

text(W / 2, 34, "Why QuickScorer cannot serve sparse-oblique splits",
     size=20, anchor="middle", weight="bold")

# ---------------------------------------------------------------- Tier A ----
text(40, 76, "A — axis-aligned: what QuickScorer requires",
     size=15, color=BLUE_D, weight="bold")

# Nodes (any tree, any depth) that test feature x3.
node_y = [96, 152, 208]
labels = [("tree 1, node a", "x₃ ≥ 0.4"),
          ("tree 2, node f", "x₃ ≥ 2.6"),
          ("tree 7, node c", "x₃ ≥ 1.1")]
for y, (who, cond) in zip(node_y, labels):
    box(40, y, 180, 44, BLUE_D, BLUE_F)
    text(52, y + 18, who, size=11, color=GRAY)
    text(52, y + 36, cond, size=14, weight="bold")

for y in node_y:
    arrow(224, y + 22, 320, 172)
text(258, 250, "compile:", size=12, color=GRAY, anchor="middle")
text(258, 264, "bucket by feature", size=12, color=GRAY, anchor="middle")

# The per-feature bucket: one sorted threshold ladder + leaf masks.
box(330, 96, 290, 156, BLUE_D, "#F7F7F5")
text(345, 118, "bucket of feature x₃", size=13, weight="bold",
     color=BLUE_D)
text(345, 134, "thresholds sorted once, offline", size=11, color=GRAY,
     style="italic")
for i, (t, m) in enumerate([("0.4", "101101"), ("1.1", "110111"),
                            ("2.6", "011111")]):
    y = 158 + 28 * i
    text(360, y, t, size=14, weight="bold")
    arrow(395, y - 5, 425, y - 5)
    text(435, y, f"leaf mask {m}", size=13)

arrow(624, 174, 690, 174)

# Scoring.
box(700, 96, 260, 156, GREEN_D, "#F2F8F4")
text(715, 120, "score one example:", size=13, weight="bold", color=GREEN_D)
text(715, 144, "• read x₃ once", size=13)
text(715, 166, "• one ascending scan, early exit", size=13)
text(715, 188, "• AND the leaf masks", size=13)
text(715, 214, "node ↔ one feature ↔ one constant:", size=12,
     color=GRAY)
text(715, 230, "this pairing makes the layout possible", size=12, color=GRAY)

# ---------------------------------------------------------------- Tier B ----
text(40, 316, "B — sparse oblique: both pillars break",
     size=15, color=ORANGE_D, weight="bold")

box(40, 336, 250, 52, ORANGE_D, ORANGE_F)
text(52, 356, "node u", size=11, color=GRAY)
text(52, 376, "0.7·x₃ − 1.2·x₁₇ ≥ 0.5",
     size=14, weight="bold")
box(40, 412, 250, 52, ORANGE_D, ORANGE_F)
text(52, 432, "node v", size=11, color=GRAY)
text(52, 452, "0.3·x₃ + 0.9·x₈ ≥ −0.2",
     size=14, weight="bold")

# Feature bucket stubs; each oblique node points at several of them.
bucket_y = {"x₃": 336, "x₈": 392, "x₁₇": 448}
for name, y in bucket_y.items():
    box(400, y, 120, 40, GRAY, "#F7F7F5", dash="4 3")
    text(460, y + 25, f"bucket {name}", size=12, anchor="middle", color=GRAY)

arrow(294, 362, 396, 356, color=ORANGE_D, dash="5 4")   # u -> x3
arrow(294, 362, 396, 468, color=ORANGE_D, dash="5 4")   # u -> x17
arrow(294, 438, 396, 366, color=ORANGE_D, dash="5 4")   # v -> x3
arrow(294, 438, 396, 412, color=ORANGE_D, dash="5 4")   # v -> x8

text(345, 408, "✗", size=30, color=RED, anchor="middle", weight="bold")
text(400, 512, "one node → several features:", size=12, color=RED)
text(400, 528, "no single home bucket", size=12, color=RED)

box(580, 336, 380, 152, ORANGE_D, "#FFFFFF")
text(595, 360, "① condition spans several features", size=13,
     weight="bold")
text(610, 378, "→ cannot be filed in a feature-major layout", size=12,
     color=GRAY)
text(595, 408, "② compared value is a node-private dot", size=13,
     weight="bold")
text(595, 426, "product Σ wᵢ·xᵢ, not a raw feature",
     size=13, weight="bold")
text(610, 444, "→ thresholds of different nodes live on different",
     size=12, color=GRAY)
text(610, 460, "axes: nothing shared to sort or scan; evaluating", size=12,
     color=GRAY)
text(610, 476, "each node separately = ordinary tree traversal", size=12,
     color=GRAY)

# ---------------------------------------------------------------- Footer ----
box(40, 560, 920, 100, GRAY, "#F7F7F5")
text(56, 586, "What YDF does today", size=13, weight="bold")
text(56, 608, "kObliqueCondition is absent from the QuickScorer condition "
     "whitelist (AllConditionsCompatibleQuickScorerExtendedModels,",
     size=12.5)
text(56, 626, "serving/decision_forest/register_engines.cc) → "
     "IsCompatible() fails → the model falls back to the generic "
     "traversal engine, which", size=12.5)
text(56, 644, "evaluates the dot product at every visited node "
     "(kNumericalObliqueProjectionIsHigher, decision_forest_serving.cc).",
     size=12.5)

add("</svg>")

with open(OUT, "w") as f:
    f.write("\n".join(out) + "\n")
print(f"wrote {OUT}")
