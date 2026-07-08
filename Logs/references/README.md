# References — orientation sheet

**Purpose:** a fast way for an agent (or a new contributor) to orient itself on the
handful of papers this project is *directly* built on — what the method is, why we
chose these systems, and which paper to open when a question comes up. This is **not**
an exhaustive bibliography; it's the short shelf of things you should actually know.

If you find yourself unsure what RGFN is, how a GFlowNet works, how the training loop is
supposed to run, or why we use the 6TD3 system, the answer is here — read the relevant
entry before reasoning further.

## Conventions

- **Cite by key, don't paraphrase.** In logs/docs write `[koziarski2024rgfn]`. The one
  authoritative citation string lives in `references.bib`.
- **PDFs are in `pdfs/<key>.pdf`** and are **git-ignored** (copyrighted + binary; keeps
  the repo clean for the code release). Open them locally by path. If one is
  missing, fetch it from the arXiv/DOI link below and name it after its key.

---

## What we're building (in one paragraph)

We fork **RGFN** (`[koziarski2024rgfn]`) as the generator and train it through the
**multi-round active-learning loop** from the original GFlowNet paper
(`[bengio2021gflownet]`, §4.3 / Alg. 1): RGFN samples molecules proportional to a fast
**proxy** reward; an expensive **oracle** (our docking-based *neosubstrate differential*,
MD later) scores only the per-round query batch; the proxy is refit on those labels;
repeat. The novel piece is **the oracle itself** — a two-tier, same-pose docking
differential that scores whether a candidate's *arm* recruits the second protein of the
ternary complex. Code: the generator lives in RGFN's `rgfn/gfns/reaction_gfn/`; our
oracle/proxy/sampler code lives in the `glue/` package and plugs into RGFN's reward
interface; configs in `configs/glue/`. See `docs/RESEARCH_CONTEXT.md` for the full picture.

---

## Method — what we're building with

### `[bengio2021gflownet]` — Flow Network based Generative Models (NeurIPS 2021)
The original **GFlowNet**. Two ideas to internalize:
1. A GFlowNet learns to *build an object step by step* and sample it with probability
   **proportional to a reward `R(x)`** — so you get many diverse high-reward samples, not
   one optimum.
2. **The training loop we use is theirs.** §4.3 ("Multi-Round Experiments") and
   **Algorithm 1** in Appendix A.5 define the active-learning loop: a learned **proxy `M`**
   (warm-started on a seed set `D_0` of true-oracle labels) is the in-loop reward, RGFN
   trains against `M(x)^β`, a query batch is sampled and scored by the expensive
   **oracle `O`**, the labels accumulate (`D_i = D̂_i ∪ D_{i-1}`), and `M` is refit on the
   full history each round. Their molecule instantiation (A.5.2) is an MPNN proxy predicting
   AutoDock scores, refit on ~200 freshly docked molecules per round — the direct template
   for our setup, with our docking differential in the role of `O`. **`O` scores enter only
   by retraining `M`, never as a direct RGFN reward.** Full pseudocode is transcribed in
   `docs/RESEARCH_CONTEXT.md` ("How the model learns"). &nbsp;`pdfs/bengio2021gflownet.pdf` · arXiv:2106.04399

### `[koziarski2024rgfn]` — RGFN: Synthesizable Molecular Generation (NeurIPS 2024)
**The paper this whole fork builds on.** RGFN = Reaction-GFlowNet: instead of growing a
molecule atom-by-atom, it assembles it through a **DAG of chemical reactions over a
building-block library**, so every generated molecule is synthesizable by construction.
The entire `rgfn/gfns/reaction_gfn/` package implements this; `data/chemistry.xlsx` is the
building-block/reaction library; our `glue/` oracles plug into its proxy/reward interface.
When extending the model, "does this match RGFN?" is answered here.
&nbsp;`pdfs/koziarski2024rgfn.pdf` · arXiv:2406.08506

### `[malkin2022trajectorybalance]` — Trajectory Balance: Improved Credit Assignment in GFlowNets (NeurIPS 2022)
**The GFlowNet training objective RGFN actually optimizes** (`configs/objectives/trajectory_balance.gin`
→ `@TrajectoryBalanceObjective`). TB learns the forward policy `P_F`, backward policy `P_B`, and a
single **global** scalar `Z = F(s0)` (the `logZ` parameter) by matching, over each complete
trajectory, `Z·∏P_F = R(x)·∏P_B` (their Eq. 13; Prop. 1 proves a global minimizer samples ∝ reward).
Crucially it learns **no per-state flow `F(s)`** — unlike flow-matching/detailed-balance — which is why
`glue/analysis/` estimates a hub's flow by forward-sampling visit count *and* recovers it from the
balance condition as `F(h)=R(x)·P_B(h|x)/P_F(x|h)` (`glue/analysis/tb_flow.py`; the two are the
`Z·∏P_F` and `R·∏P_B` sides of the TB loss, so their agreement is a training-quality check).
&nbsp;`pdfs/malkin2022trajectorybalance.pdf` · arXiv:2201.13259

---

## Baselines — what we compare against

### `[seo2024rxnflow]` — RxnFlow: Generative Flows on Synthetic Pathway for Drug Design (2024, ICLR 2025)
Our primary **synthesis-aware** baseline — the synthesizable peer to RGFN (FragGFN
is the *non*-synthesizable foil). Like RGFN, RxnFlow is a GFlowNet that assembles
molecules along a **synthetic pathway** — picking a building block, then applying a
**reaction template** to a chosen reactant — so every sampled molecule carries a
forward-synthesis route. Its headline contribution is an **action-space subsampling**
trick that lets it learn over a huge action space (~1.2M building blocks × 71
reaction templates) without retraining when the library changes. It is built on
Recursion's `gflownet` (bundled, v0.2.0) — the *same* base as our FragGFN entrant —
so it drops into the same two-env pattern. We run it through the **same**
active-learning loop, the **same** 6TD3 docking oracle, the **same** seed/budget/β,
and the **same** proxy `M` as the RGFN entrant, so the comparison isolates the
generator. The standard candidate dataset records its routes (`has_route=1`,
`routes.jsonl`) — the differentiator vs. FragGFN. Heavy upstream code is installed
via `external/setup_rxnflow.sh` (not vendored); the thin adapter lives in
`validation/generators/rxnflow/`. &nbsp;`pdfs/seo2024rxnflow.pdf` · arXiv:2410.04542

### `[gainski2025scent]` — SCENT: Scalable and Cost-Efficient de Novo Template-Based Molecular Generation (2025)
Our **cost-aware** baseline — and the closest relative of all of them: SCENT is a
**fork of RGFN from the same lab** (its package is literally named `rgfn`; same
stack — py3.11/torch2.3/dgl/gin — same `rgfn.api`, same `train.py --cfg ….gin`),
so RGFN is one of *its* own baselines. It keeps RGFN's reaction-template,
synthesizable action space and adds three things on top: **Recursive Cost Guidance**
(auxiliary models that estimate synthesis cost from building-block prices + reaction
yields, steering the backward policy toward cheap routes), an **Exploitation Penalty**
(visitation-count term that keeps cost guidance from collapsing diversity), and a
**Dynamic Library** (promotes high-value intermediates to building blocks, enabling
tree-structured routes). Because the package name `rgfn` collides with ours, it runs
in its **own `scent` conda env** and reaches the shared 6TD3 oracle across the env
boundary via `scripts/score_batch.py` — the *same* two-env bridge pattern as FragGFN
/ RxnFlow, just forced by a namespace clash rather than a version clash. We run it
through the **same** active-learning loop, oracle, seed/budget/β, and proxy `M` as the
RGFN entrant, so the comparison isolates what SCENT's cost-awareness buys. It is
synthesizable (`has_route=1`, `routes.jsonl`). Heavy upstream code installed via
`external/setup_scent.sh` (not vendored); thin adapter in
`validation/generators/scent/`. &nbsp;`pdfs/gainski2025scent.pdf` · arXiv:2506.19865

---

## Evaluation — synthesizability metrics

### `[genheden2020aizynth]` — AiZynthFinder: a fast, robust retrosynthesis tool
The retrosynthesis engine behind the **synthesizability metric we report on every
entrant** (`validation/harness/synthesizability.py`). Given a target SMILES it runs a
Monte-Carlo-tree search over USPTO reaction templates back toward a **building-block
stock** (we use the standard public ZINC in-stock set); a molecule is **"solved"** iff a
full route to in-stock precursors is found. The headline number is the **fraction
solved** ("AiZynth success rate") — exactly the `AiZynth` column in `[koziarski2024rgfn]`
Table 1 (RGFN ≈ 0.56) and `[gainski2025scent]` Table 1 (up to ≈ 0.75), and RxnFlow's
"Synthesizability %" (`[seo2024rxnflow]`). Note these papers stress it is **noisy and
conservative** (RGFN molecules an expert confirmed synthesizable score below 1.0), so it
is a *post-hoc validation* metric, never an in-loop reward. Installed in its own
`aizynth` conda env via `external/setup_aizynthfinder.sh`. &nbsp;DOI:10.1186/s13321-020-00472-1

### `[ertl2009sascore]` — Synthetic Accessibility (SA) score
The cheap, RDKit-native companion to AiZynth that the same papers also report (a 1 = easy
… 10 = hard heuristic from fragment contributions + complexity penalties). We compute it
alongside the AiZynth verdict in the same evaluator. &nbsp;DOI:10.1186/1758-2946-1-8

---

## Domain — systems, glue design & evaluation

### `[koziarski2024rgfn]` is method; these explain the *chemistry* we're scoring.

### `[bengeoffrey2025molde]` — Molecular Glue-Design-Evaluator (ACS Omega 2025)
In-silico method for **designing and scoring molecular glues**. Consult this when
reasoning about glue design or oracle scoring choices — and note it's the reference
behind our choice of the **6TD3** system as the testbed (`Logs/002_6td3-cr8-validation-and-discrimination.md`).
&nbsp;`pdfs/bengeoffrey2025molde.pdf` · doi:10.1021/acsomega.4c08049

### `[slabicki2020cr8]` — CR8 is a molecular glue degrader of cyclin K (Nature 2020)
Source of the **6TD3** system: DDB1·CDK12–cyclinK·CR8 ternary complex — our **validated
oracle** (78-pp separation on the neosubstrate differential). Cited in logs 002, 003, 005.
&nbsp;doi:10.1038/s41586-020-2133-z &nbsp;*(no PDF — paywalled; drop in if obtained)*

### `[matyskiela2016cc885]` — Cereblon modulator recruits GSPT1 (Nature 2016)
Source of the **5HXB** system: CRBN·DDB1·GSPT1·CC-885 — the **ceiling-hit** system where
docking can't separate real glues from decoys. Cited in logs 001, 003, 005.
&nbsp;doi:10.1038/nature18611 &nbsp;*(no PDF — paywalled; drop in if obtained)*
> ⚠️ Logs 001 & 003 cite this as "Science 2016" — it is **Nature 535:252–257 (2016)**.
> Fix when those logs are next touched.

---

## Adding a paper

Keep this sheet short — add a paper only if work genuinely builds on it.
1. Drop the PDF in `pdfs/` named `<citekey>.pdf` (won't be committed — fine).
2. Add the BibTeX entry to `references.bib` (key = `<firstauthor><year><tag>`).
3. Add a 2–4 line entry here: what it is, why we care, and a pointer if useful.
4. Cite it by key from logs/docs — never restate the citation inline.
