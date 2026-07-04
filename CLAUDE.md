# Oblique RF performance research — auto-loaded context

All work in this repo targets one thing: making `ProjectionEvaluator::Evaluate()`
(ApplyProjection) in `yggdrasil_decision_forests/learner/decision_tree/oblique.cc`
faster, under the workflow below. These are the standing context — follow them.

`AGENTS.md` is the workflow; `oblique_context/OBLIQUE_CONTEXT.md` is the lean code-map
core (scope, call map, the hot function, driver skeleton, invariants, measured facts).
Deeper detail is sharded into sibling files under `oblique_context/` (split finders,
tree growers, kernel variants, dataset/sampling source, build/measure) and read on
demand via the router table in the core — only the core is auto-loaded here.

@AGENTS.md
@oblique_context/OBLIQUE_CONTEXT.md
