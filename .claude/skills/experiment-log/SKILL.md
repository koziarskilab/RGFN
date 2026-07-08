---
name: experiment-log
description: Write structured experiment logs for research projects. Use at the START of any computational experiment (docking runs, MD simulations, oracle validation, pipeline tests, discrimination checks, compute benchmarks) to create a log stub capturing the goal and context. Use at the END of an experiment to write up the full results, file inventory, and connection to the publication. Also trigger when the user says "start a log", "log this experiment", "write up the results", "document what we did", "wrap this up", or "create an experiment entry". If you are at the conclusion of an experimental session and have results, files, and commands to document — write the log without waiting to be asked.
---

# Experiment Log Skill

Each log has two layers that serve different readers:

**Story layer** — Question, Context & Summary, Answer, Relevance to our Publication, Next Experiments. Written for a smart person who isn't an expert on this specific project. Prioritize being understandable and engaging over being technically precise. Tell the story of what happened and why it matters.

**Audit layer** — everything from `# Re-creation` onwards (Relevant Files, Relevant Versions, Relevant Resources, Method, Results). Written for reproducibility. Technical accuracy matters here. Scientific terms, file paths, and exact commands belong in this layer.

If you're unsure which layer a detail belongs in: if it helps a reader understand the experiment's purpose and outcome → Story. If it helps someone reproduce or audit the experiment → Audit.

In the story layer: stick to facts stated explicitly in the conversation or in `RESEARCH_CONTEXT.md`. Don't infer technical explanations for *why* something works or fails — just state what happened and what it means. If you're uncertain about a technical detail, leave it out rather than guessing.

---

## Step 0: Orient yourself

Read two files before writing anything:

1. **`Logs/RESEARCH_CONTEXT.md`** — the paper's central claim, current project status, what journals will want, and key terminology. This tells you what the experiment is *for*. If it doesn't exist, ask the user to describe the research context.

2. **`Logs/README.md`** — the index of past experiments. Reading it tells you what's established so you can say things like "as shown in entry 002" accurately.

---

## Step 1: Determine mode

**START mode** — experiment hasn't run yet. Create a stub.
**END mode** — experiment is complete. Fill everything in, or find a stub and complete it.

To find an existing stub: `grep -rl "\[TODO" Logs/ HistoricLogs/ 2>/dev/null`

---

## Step 2: Find the next log number

```bash
ls Logs/*.md 2>/dev/null | grep -Eo '/[0-9]+_' | grep -Eo '[0-9]+' | sort -n | tail -1
```

Increment by 1, zero-pad to 3 digits: `005`, `006`, etc. If the user specifies a different output directory (e.g., "write to HistoricLogs"), use that directory but the same numbering.

---

## Step 3: Write the log

**Filename:** `NNN_kebab-case-description.md` (3-6 words, e.g., `006_brd4-vhl-generalization-test`)

### Header

```
# [System] — [brief description]
**Date:** YYYY-MM-DD, ~[rough time, e.g., "10am"]
```

Do NOT include the log number in the title — it's already in the filename.

---

### Question

*(Story layer. Write this last if needed — it's the hardest to get right.)*

One sentence. The test: can someone skim this and immediately understand what we were trying to find out? No acronyms, no method names. Just the question.

Don't add implications or "could this also mean X?" — those belong in Context. Keep it clean.

> **Good:** "Can our scoring method tell real molecular glues apart from randomly generated molecules across multiple protein systems?"
> **Good:** "Can we use the stability of the protein complex over a short simulation to score molecular glue candidates better than a static docking score can?"
> **Avoid:** "Does the DDB1 neosubstrate differential discriminate in the 7ABC system with exhaustiveness 16, and could this replace or improve on our current oracle pipeline?"

---

### Context & Summary

*(Story layer. Explain numbers in plain terms. Name specific venues. Don't pre-explain failure scenarios.)*

Two parts:

**Context** — Why is this experiment necessary? What earlier result or open question makes this the next logical step? Reference prior log entries by number. When citing numbers from prior entries, explain what they mean in plain terms — don't assume the reader remembers. Example: "Entry 002 showed we can tell real glues from randomly generated molecules with about 78% accuracy for the CDK12-DDB1 system." Keep to 3-5 sentences.

**Summary** — What are we going to do? One short paragraph in plain English. If introducing a new control or comparison group, describe its PURPOSE ("randomly generated molecules that act as our negative control") before using a shorthand term for it. Don't explain what we'll do if things go wrong — focus on what we're trying to achieve.

---

### Answer

*(Story layer. Write this before writing Results — it forces clarity. Focus on meaning, not numbers.)*

2-3 sentences on what the results *mean*. What can we now say with confidence? What did this experiment add to our understanding? Numbers belong in Results; the interpretation belongs here.

---

### Relevance to our Publication

*(Story layer. Be specific about which venue and which reviewer concern this addresses.)*

How does this experiment help us publish? Name the specific venue (NeurIPS, Nature, etc.) and the specific thing reviewers will look for. Example: "NeurIPS reviewers will ask whether our method works beyond a single protein system — this entry answers that directly with a second validated system." One short paragraph.

---

### Next Experiments

*(Story layer. Fit everything into the publication goal. Avoid jargon — describe purpose, not method.)*

Two subsections:

**Refining for publication** — What will reviewers still want? Things that make the existing result more airtight: additional trials, cleaner figures, ablations showing a design choice is necessary. Ablation experiments belong here, not in Next Steps.

**Next steps in project** — What experiments come next in the project pipeline? Frame in terms of the publication goal: "Run RGFN with each validated oracle to show the model can generate good candidates for multiple systems."

---

### `# Re-creation`

This heading is **required** — it marks where the audit layer begins. Everything below this line is for reproducibility and auditing, not for storytelling.

---

### Relevant Files

*(Audit layer. Focus on role and narrative context — WHY this file, not just WHAT it is.)*

State the root directory once if most files share a prefix. Use these path conventions:
- `./path` — relative to the project repository root
- `/path` — absolute path outside the repository (scratch dirs, SLURM logs)
- `path` — relative to the stated root

For **scripts**: path + one-line description of what it does.

For **non-script files** (models, datasets, results): path + description that explains its role in the pipeline. Focus on WHY this file is used and what it represents in context — e.g., "CDK12 receptor alone (Tier 1), used to isolate E3-pocket binding from neosubstrate cooperativity." You don't need to hedge ("believed to contain") — just describe what the file is.

Categories (only include what applies, ordered by pipeline stage):
- **Scripts**
- **Models**
- **Datasets**
- **Results**
- **Job Logs**

---

### Relevant Versions

*(Audit layer.)*

Run:
```bash
git log --oneline -5
git status --short
```

**If relevant files are committed:** paste the most recent relevant commit hash and message.

**If relevant files are NOT yet committed:** tell the user exactly which files need to be committed, leave a `[TODO — add commit hash after pushing]` placeholder, and explicitly ask: "Can you commit the experiment files? Once you do, let me know and I'll update this section with the commit hash." Then wait — if the user confirms, update the log.

---

### Relevant Resources

*(Audit layer.)*

Sources consulted when designing or running the experiment. Two subsections:

**Sources** — papers, databases, PDB entries (with DOIs or URLs where available)

**Packages** — tools and libraries used, with the specific file(s) in this project that use them

---

### Method

*(Audit layer. Big commands only — not debugging steps or package installs.)*

Numbered steps: what was run, what it operated on, what it produced.

---

### Results

*(Audit layer. Numbers that support the Answer section. Use tables for comparisons.)*

Label clearly: n counts, metric names, units. If pulling numbers from a prior log entry (not generated in this experiment), note which entry they came from — e.g., "(from entry 002, job 69271)" — so future readers know not to re-derive them here.

---

## Step 4: Update the README index

Add one row to `Logs/README.md` (or the equivalent in the output directory):

```
| [NNN](NNN_filename.md) | YYYY-MM-DD | Brief title | One-sentence verdict |
```

The verdict should be punchy: someone skimming the index should immediately know the key finding.

---

## Quality check

- [ ] Title has no log number (it's in the filename)
- [ ] Date includes a rough time
- [ ] Question is one clean sentence, no jargon, no implications clause
- [ ] Context explains what numbers *mean*, not just what they are
- [ ] `# Re-creation` heading present
- [ ] No `[TODO]` markers in completed END-mode sections (except intentional commit-hash placeholder)
- [ ] README row added
