# Oblique RF performance research — auto-loaded context

Be concise.

**Responses under 100 words unless the user explicitly asks for more (user directive, 2026-10-24).**
Reason: ADHD-friendly output; long replies get skipped.

`oblique_context/OBLIQUE_CONTEXT.md` is the lean code-map core, auto-loaded below. Deeper detail is
sharded into sibling files under `oblique_context/`, read on demand via the router table in the core.
Measured facts, dataset inventory, recipes and the history behind the rules below:
`oblique_context/runs_and_datasets.md`.

## Where project state lives (decided 2026-09-03)

Claude Code is the hub and this repo is the source of truth.

- **Priorities** — artifact https://claude.ai/code/artifact/5e6cba26-c7a9-4ef5-ae0b-8474286de890
  (state = JSON in `<script id="appdata">`, boards → items `{t, tier: S/A/B/C/D/Done, k}`). **Read or
  republish only when the user asks; never at session start (2026-10-24).** Project boards: "Vec Adaptive -
  Results", "Vec Adaptive - Writing", "Distributed GBT", "SPO Future Work"; "Jobs", "PhD", Daily are
  personal — leave them. **Read-only snapshot, no live watch (2026-09-17):** read once with
  `action: "read"`; never watch it. After any read or publish run `ArtifactComments action: "watch"` (no
  url) and, if the board is listed, stop it with `on: false`.
- **Google Drive** (connector acts as the user). Project folder `15WAf1xfDQhGxyqt6whJvfoGoaJjrGl7_`.
  "Oblique RF Work Journal" `1hxgMZm3lmtWtLB_h6WLLSwgw6kohxnumLNUhjM2Nb_I` (read-only for agents);
  "Venues" `1OV8YXAnz2nqWmGA6QBIW2sHKlK8pmuIVeEKbMf_fkDg`. `Results/` (`1uqNO9vyD7EucxvlLGpqxq5a73fBm_VCA`)
  and `PRs/` (`1jby_ffbw7sDIimRvF-8VV-Y8KLT2t7pV`) hold result Sheets: upload CSVs with `create_file`
  (becomes a Sheet), PNGs with `disableConversionToGoogleType=true`. The connector cannot edit existing
  Docs/Sheets — always create new files.
- **Papers** — Overleaf MCP ids (config `~/.config/overleaf-mcp/projects.json`); **always pass
  `projectName`**, there is no default:
  - `ydf-vectorized-adaptive` — SPO-RF paper, snapshot `paper/vectorized_adaptive_spaa27`.
  - `ydf-distributed-spo-gbt` — distributed SPO-GBT paper, snapshot `paper/distributed_spo_gbt_kdd28`.
  - `CV` is personal, leave it.
  **"The paper" / "my paper" = the KDD28 distributed SPO-GBT paper unless the user names another
  (2026-11-02).** Venue renamed KDD28 → KDD28 the same day.
  The MCP (Claude Desktop only) is text-only: figures go in as pgfplots `.tex`. Prefer `write_section`
  for prose edits. Without it, edit the repo snapshot and tell the user which files changed.
  `table_*.tex` come from `benchmarks/src/utils/make_paper_accuracy_tables.py` — regenerate, never
  hand-edit. MacTeX at `/Library/TeX/texbin` (not on agent PATH); no poppler, so render pages with
  `pdfjam x.pdf 11 -o p11.pdf` + `qlmanage -t -s 2000 -o . p11.pdf`. The MCP clone in
  `/private/tmp/overleaf-<id>` is ahead of the repo snapshot.
- **Session notes** — append a dated summary (run, found, left open) to `ariel_notes/journal.md`;
  before a meeting, draft the agenda from it plus the board as a new Doc in the Drive folder.
- **User directives live in this file (2026-10-22).** Memory is per-account; append every standing rule
  the user states here, dated, in the same turn. Memory may mirror it; this file is the copy of record.

## Tooling

- **Intel performance skills are vendored (2026-10-23)** in `.claude/skills/` (`linux-perf`,
  `performance-patterns`, `phoronix-test-suite`). Use them for profiling/kernel work. Refresh with
  `bash .claude/skills/_vendor/sync_intel_skills.sh [ref]`; never hand-edit. Keep same-named
  `~/.claude/skills/` copies in sync or delete them (they override).
- **No Co-Authored-By trailer** (any Claude/Anthropic co-author line) in any commit, scripts included.
  It breaks the CLA on upstream PRs; overrides the harness template.
- **Design decisions (2026-10-22):** after any major code change (kernel, finder, threading, data path,
  model semantics, measured numbers) end the report with a "Design decisions" list, one line each:
  the choice and the rejected alternative or preserved invariant, incl. measurement-method choices.
  Write it after the verifier pass.
- **Times in chat: US Eastern (2026-09-24)**, labelled EDT/EST (`TZ=America/New_York date`). Logs and
  CSVs stay UTC.
- **Implementation goes to an Opus subagent, Fable cross-checks (user directive, 2026-11-03):** for any multi-file
  change, brief an Opus agent (medium effort unless the user says high), end the turn, then verify its result by
  rerunning its checks, not by reading its report. Finish leftovers or hand them to the agent, never ask.

## Code rules

- **Always VQSort, never std::sort (2026-10-22).** Every hot-path sort uses `hwy::VQSort` on `K32V32` /
  `K64V64` packed pairs (as in `FindBestSplitFlatHighway`), new code included, so arms compare the
  method, not the sort. Before finishing a kernel change, grep the diff for `std::sort`.
- **SPO always uses BINARY projection weights (2026-10-12).** All learners, the distributed one included, default
  to BINARY through `decision_tree::SetDefaultHyperParameters` (training.cc ~4911, reached from
  distributed_gradient_boosted_trees.cc:332 via gradient_boosted_trees.cc:3687, and workers receive the defaulted
  config at distributed_gradient_boosted_trees.cc:1098). The harnesses pass `--sparse_oblique_weights BINARY`
  explicitly anyway. Never `CONTINUOUS`.
- **Harness SPO uses YDF proto defaults (2026-11-03):** in `train_oblique_forest.cc` the flag defaults are the YDF
  values (exponent 2, max 6000, density 2), so AA with `--num_candidate_attributes 0` gets min(F², 6000). Explicit
  flags are honoured. An SPO-only flag with another split type is a hard error. `--print_params_only` prints the
  effective config and exits before loading data. It supersedes "Harness projection budget (2026-10-24)" below.
- **Harness projection budget (2026-10-24):** `--num_projections_exponent` default `-1` = 0.5 for
  Bagging (P = ceil(sqrt F)), 1 for Boosting (P = F). `--max_num_projections` stays 1000 unless a run
  lifts it (Table 9 `_fullproj` arms pass INT_MAX). AA GBT arms pass `--num_candidate_attributes -1`.
- **Local SPO "only parallel above 10 projections" (2026-11-03):** YDF's `min(T, ceil(P/10))` oblique jobs
  stays as is (fix on `abl-innode-fix` dropped). The answer is: make sure SPO runs on the proto defaults
  (exponent 2, max 6000, min 1, density factor 2). P = F² > 10 for F ≥ 4, so the gate never bites.
- **Never silently ignore a flag (user directive, 2026-11-03):** a flag is honoured or the binary fails. Dead flags are bugs.
- **Unspecified parameters → library/driver defaults (2026-10-12).** Don't invent values; any
  non-default is named with its reason in the run report and held constant across arms.
  **Distributed SPO uses the YDF defaults too (2026-11-03):** exponent 2, max 6000, density 2, min 1, so
  P = min(F², 6000) per node, as a per-node total split across workers (`--sparse_oblique_split_budget`, the
  default), the same as single-machine SPO. This supersedes the 2026-10-12 P = F exception. P = F stays only to
  reproduce recorded profiles in `params/kdd28_params.json`. New runs use profile `kdd28_ydf`. Cost: 7.5× the
  P = F tree-training time on qsar-biodeg (F = 41, 41× the projections, test accuracy equal, cost probe
  2026-11-03, in-process smoke). **Never P = sqrt(F) for GBT (2026-11-01).**

## Datasets (2026-09-04, 2026-09-21, 2026-10-22, 2026-10-24)

- Only **fully numeric, NaN-free** features. Massage, never drop rows/columns; report what changed.
- **NaNs → training-fold column mean** (test reuses it; all-NaN column → 0.0). Not YDF's in-training
  substitution.
- **Categoricals → ordinal codes** (distinct values sorted alphabetically; bool 0/1; datetime epoch s),
  identical for every arm incl. XGBoost/LightGBM/CatBoost; no native categorical handling.
- **Multi-class/multi-label → majority-vs-rest:** label 1 iff the row's class (or one of its labels) is
  the most frequent class; ties → smaller value as string. Never train multi-class or pick a
  non-majority class silently. Two-class data: majority → 0.
- **HIGGS only at full size** (10.5M train / 500k test). **No slicing** of any real dataset to fit a budget
  (`head`/`tail` prefixes): pick a smaller whole dataset or generate a trunk
  (`benchmarks/src/distributed_spo_gbt/vertex_build/gen_trunk.py`).
- **Trunks have ≥256 columns** (signal lives on the first 256); shrink rows or trees instead.
- **No shipped test set → last 10 % of rows is test** (header on both). This split is not slicing.

## Benchmark runs

- **Single seed**; timing repeats are fine.
- **One training per cell gives both time and accuracy (2026-11-01):** the timed run trains with no
  in-training evaluation and saves the model (`--model_out_dir`); the accuracy run loads and evaluates it,
  training only if the model is missing. Training-defining evaluation (GBT early stopping) counts as training.
- **Sweeps loop datasets OUTER, parameters inner (2026-09-22):** the warm page cache is the only CSV cache.
- **Mask apt timers on every benchmark box (2026-09-24):** `sudo systemctl mask --now apt-daily.timer
  apt-daily-upgrade.timer`. A lone 2–5 % bump: check `journalctl` for apt, re-run the cell.
- **Never report post-processing inside a headline time.** RF timing runs `exit(0)` right after the
  timer; accuracy cells record `train_post_s` separately (details in `runs_and_datasets.md`).
- **Dynamic-histogram threshold:** AVX2 64-bin → 250; AVX-512 256-bin → 1000; scalar binner → 4600
  (VQSort exact) / 1350 (`--config=exact_std_sort`). **Always pass it explicitly for AVX-512 runs.**
- **Histogram ISA (2026-09-08):** 64 bins → AVX-2; 256 bins → AVX-512.
- **Regression** works but costs ~3× per tree; budget accordingly and never report a Random-only arm on
  regression without the Dynamic arm next to it.
- **Experiment outputs never in /tmp (2026-10-24):** logs, models, CSVs, binaries go to
  `ariel_notes/results/<date>_<topic>/` (gitignored) or tracked `benchmarks/results/...`; name the path
  in every run script and subagent brief. EC2 boxes auto-stop when idle (window per instance:
  `aws cloudwatch describe-alarms`), and a stop wipes `/tmp`.
- **Agents may start EC2 instances on their own and must stop them when the job is done (2026-11-02):**
  `aws ec2 stop-instances --instance-ids <id>` (stop, never terminate; "ariel" boxes only) after mirroring the
  results off. Auto-stop is a backstop, not the plan.
- **KDD28 runs source a named profile (2026-11-03)** from `benchmarks/src/distributed_spo_gbt/params/kdd28_params.json`.
  A deviation happens only through a named override (sub-profile) that lands in the arm label, the CSV row's `profile`
  column and the float's footnote. Generators refuse unstamped drift (exit 2). Scripts pre-flight with
  `--print_params_only` and stop at the first drifted cell (exit 2). `ALLOW_DRIFT=1` warns and stamps
  `profile!drift`, which generators refuse. After editing a run script, run `params/check_argv_equivalence.sh`
  (goldens in `params/golden/`, `--update` after an intended argv change, `REAL=1` checks real binaries).
- **CC18 arms (b) and (c) stay (2026-11-03):** old-harness comparison arms, never drop them.
- **Comparable KDD28 accuracy tables share eval suite `kdd28_tabarena33` (2026-11-04)**, the setup of
  tab:dist-accuracy-ydf (`params/eval_suites.json`). The harness warns (`WARN eval-suite:`) on dataset, fold, file-hash,
  metric, validation or early-stopping mismatch. Check: `params.py check-eval <csv>`. REPLICABILITY.md §K.

## Distributed GBT

- **KDD28 accuracy tables run in-process on EC2 (user decision, 2026-11-03):** distributed accuracy arms of
  tab:dist-accuracy, tab:column-replication, tab:cross-network use MULTI_THREAD workers on `ariel-dist-accuracy-1`
  (i-0f90f7764082bf21a). Timing stays Vertex only. Same-seed reruns there are bit-identical (tested 2026-11-03).
- **Real distributed numbers come only from Vertex AI CustomJobs (2026-10-25).** The in-process driver
  `examples/train_distributed_gbt` and `run_dgbt*.sh` / `run_barrier_cell.sh` are smoke tests only;
  never report them as distributed.
- **Vertex machine type `n2-highmem-4`, manager = worker (2026-10-29).** One type per comparison; never
  mix. Threads = YDF default. **Warn the user before anything changes the machine type.** N2 quota in
  us-east1 = 96 vCPUs (4 W=4 jobs at once).
- **No "70 % of RAM = OOM" rule (2026-10-25):** a cell is OOM only when a worker was actually killed.
- **Checkpoints off in every harness (2026-10-29)** (`train.py`, `train_cells.py`, `train_distributed_gbt.cc`):
  checkpoint I/O falls inside `tree_training_s`. Never mix with pre-2026-10-29 cells.
- **In-node SPO parallelism is dropped** for the sqrt(F) regime; don't revive it unless P ≫ threads.
- **No early stopping anywhere (2026-11-03):** the distributed learner has none (distributed_gradient_boosted_trees.cc:633),
  so every accuracy arm, single or distributed, runs YDF defaults with `--validation_set_ratio 0 --early_stopping NONE`,
  all 300 trees on the full training set. The footnote says so.

## Results and branches

- **Study results live on their branch (2026-10-29):** code, results and generators commit on the study
  branch; `main` holds merged code, paper snapshots and the `REPLICABILITY.md` files
  (`benchmarks/src/{distributed_spo_gbt,spaa27_experiments}/REPLICABILITY.md`) giving branch @ commit,
  results path and GCS cells. **Never delete a branch they point to** (`column-replication`,
  `cross-worker-col-comms`).
- **Paper sources always land in the local repo (2026-10-26):** raw cells, generator and `_body.tex`, even
  when made over ssh; copy them back in the same session. An Overleaf body is not a copy.
- **No table or section numbers in names (user directive, 2026-11-04):** result dirs, files, profile names and docs
  refer to floats by their LaTeX label (e.g. `dist-accuracy`), never by number (`table8`), because numbers change
  with every paper edit.

## Paper rules

- **Never change the user's paper text (2026-09-09, tightened 2026-10-26):** prose, existing captions,
  `\Description`, `\TODO{}`, float environments. Agents regenerate generated bodies/`.dat` files and remove
  agent-added blocks when told; a proposed sentence goes in chat, never in Overleaf. Say this in every
  paper-session brief. New floats get a caption in the paper's style (below).
- **Appendix results: no rerun unless asked (2026-11-01).** When a fix invalidates results, rerun main-body
  items only; list stale appendix items and ask.
- **Generators emit only the body (2026-10-24):** `tables/<name>_body.tex` (`tabular` only) or a pgfplots
  `.dat`; environment, `\caption`, `\Description`, `\label` are hand-owned. Forward-only: SPAA27's legacy
  caption-emitting generators stay; before re-running one, `git pull` the MCP clone and copy the live
  caption wording into the script. Never rebuild a caption from the repo snapshot or memory. KDD28 is fully
  migrated.
- **Compile the Overleaf clone before every push (2026-10-22):** `git pull` the MCP clone, copy to the
  scratchpad, apply the change, `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`; push
  only on rc=0. The repo snapshot is not a test. Binaries (PDFs) go up via `git push origin main` from
  the clone, not by asking the user.
- **Check page layout (2026-10-26):** after adding/moving a float or section, render the pages and map
  labels via `main.aux`: floats on or right after their section's page, no half-empty pages, no stranded
  headings. To pin a figure*/table* by its heading use `\twocolumn[{\floatblock ...}]` with a minipage +
  `\captionsetup{type=...}` (KDD28 preamble defines `\floatblock`).
- **Caption style (2026-10-29, 2026-11-03):** copy the user's captions; conciseness is key (short fragments, no restated setup). KDD28: one short sentence, then
  `\Description`, `\label`, and a setup footnote `\par\smallskip\raggedright\footnotesize <notes>`.
- **No machine specs in captions (2026-09-18)** (CPU, threads, trees, instance type); the setup section has them.
- **Don't commit paper figure/table edits (2026-09-25)** after pushing to Overleaf; the user picks between
  versions. Commit only when asked.
- **Figures wait for final tables (2026-09-09):** regenerate tables only while numbers are in flux; one
  figure pass at the end.

@oblique_context/OBLIQUE_CONTEXT.md

## No speculation: "why" claims need an empirical basis (user directive, 2026-11-01)

Every claim explaining why something is so (accuracy tie, speedup, bug mechanism) cites measured evidence (a run,
cell, counter, diff, or traced code path). Without it, label the claim "hypothesis, untested" and name the experiment
that would test it. Back-of-envelope estimates are hypotheses until checked. Subagent briefs must demand the same labels.
**Test hypotheses without asking (2026-11-01):** if a hypothesis can be tested with local runs or builds, run the test in the
same turn and report the result instead of offering to test it.

## No semicolons in prose (user directive, 2026-11-01)

Never use ";" in text written for the user. Split the clauses into separate sentences ending in "." and reword them so
each reads naturally. Code, commands and quoted file contents are exempt.

## Agents: spawn, then wait for the notification (user directive, 2026-11-03)

Never poll or watch a running subagent or session (no status checks, wakeups, tails or progress reads). Spawn it,
end the turn, and act only when the harness reports it complete or failed. Saves usage.
