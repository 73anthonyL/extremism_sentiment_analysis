# The research loop

This document describes how a candidate technique moves from an idea to a
registered result, and what the `research_loop/` directory stores along the way.

`docs/EXPERIMENTS.md` defines the protocol a single experiment must follow. This
document covers the layer above it: how experiments are sequenced, how the test
split is rationed across them, and who is allowed to declare a winner.

## Why the loop exists

The test split is 450 rows. Detecting a difference at McNemar exact
significance needs roughly +3 accuracy points, about 14 rows. That is a hard
statistical ceiling, and it has two consequences that shape everything below:

1. Most comparisons this project can run will come back `INCONCLUSIVE`. That is
   a real, publishable outcome, not a failed experiment.
2. Repeatedly peeking at the test split until something looks good would
   manufacture a winner out of noise. So test evaluations are rationed,
   recorded, and paid for with a rising significance bar.

## Directory layout

```text
research_loop/
├── STATE.md            # current cycle, stage, blockers, and history
├── registry.json       # the current champion; written only by ledger.py promote
├── prereg/             # one preregistration YAML per cycle
├── probs/              # sanitized probability artifacts from completed runs
├── decisions/          # adjudicated verdicts, one per comparison
├── cycles/             # per-cycle working notes
├── val_log.jsonl       # declared validation looks
└── test_ledger.jsonl   # hash-chained record of every test unlock
```

`STATE.md` is the entry point. Read it before starting any work in this
repository; it records which cycle is live, which stage it is in, and what it is
blocked on.

## Cycle stages

A cycle runs through seven stages. The `/research-cycle` skill sequences them
and stops at the three human gates.

| Stage | What happens | Gate |
|---|---|---|
| audit | Verify protocol invariants and artifact integrity before anything new is built. | |
| propose | Design one candidate technique and write its preregistration. | **Human gate 1: approve hypothesis** |
| build | Author the Kaggle notebook from the approved preregistration. | **Human gate 2: run the GPU job** |
| integrate | Derive `results_summary/<TECHNIQUE>/` from the returned probability artifact. | |
| decide | Run `tools/compare_techniques.py` and record its verdict. | |
| docs | Regenerate rendered tables; update prose that no renderer covers. | |
| review | Check every claim for overclaiming against the recorded verdict. | **Human gate 3: commit** |

The gates are where a human decides. Nothing between them decides anything on
its own — in particular, no agent and no notebook may declare a technique
better than another.

## Preregistration

Before a run, a cycle writes `research_loop/prereg/<TECHNIQUE>.yaml` fixing at
minimum:

* the hypothesis and the comparator it is measured against
* the decision rule, including the threshold policy, prespecified before the run
* which challengers may replace the primary rule, and the margin each must clear
* the number of validation looks the run is allowed
* the validation gate that must pass before the test split is unlocked at all
* the expected outcome, stated honestly — usually `INCONCLUSIVE`

The prereg SHA is passed to `tools/compare_techniques.py` at adjudication, so a
verdict is always tied to the design that was fixed before the numbers existed.

A validation gate is not a formality. A candidate that cannot clear a
validation bar has no business spending a test unlock, and a run that falls back
to reproducing an existing model should refuse the unlock outright rather than
spend one re-measuring something already measured.

## Probability artifacts and derived results

A completed run does not report metrics. It exports probabilities, and the
metrics are derived from them.

The artifact contract is exactly four columns — `row_id`, `split`, `y_true`,
`y_prob` — with no text-bearing columns, no duplicate row ids, and all
probabilities finite and within `[0, 1]`. Notebook 11 contains a working
implementation of the export, including the assertions that enforce the
contract.

```bash
# after copying probs/*.csv and probs/*__meta.json into research_loop/probs/
python3 tools/eval_from_probs.py --technique <TECHNIQUE> --threshold <selected>
```

This builds the whole result folder. The threshold comes from the run's
metadata; `eval_from_probs.py` does not choose one, and it hard-errors on any
attempt to select a threshold on test-split data. `ablation_results.csv` is the
one required file it cannot derive, so that is added by hand from the run's
configuration comparison.

The payoff is that a number in a results table can only change because a
committed artifact changed. Anything transcribed by hand from a notebook is
outside this guarantee, which is why such numbers are kept out of the rendered
tables and labelled as unregistered wherever they appear in prose.

## The test ledger and the Holm family

`research_loop/test_ledger.jsonl` is a hash-chained log of every test
evaluation. Each entry links to the previous one, so an entry cannot be
backdated or removed without breaking the chain, which `tools/ledger.py verify`
checks.

Each entry also raises the multiple-comparison family size that every future
candidate must clear under Holm correction. This is the price of looking at the
test split: the more times it has been read, the stronger a new result must be
to count.

```bash
python3 tools/ledger.py verify
```

The ledger is currently **empty** — it was lost along with `research_loop/` and
`tools/` before the 2026-07-29 reconstruction, and only evaluations made through
the tool may be recorded. Backdated entries must not be fabricated. Meanwhile
the true family is larger than the ledger length, so **`--family-size` must be
passed explicitly to every comparison**. `research_loop/STATE.md` carries the
current count and its derivation.

## Adjudication

```bash
python3 tools/compare_techniques.py \
  --candidate <TECHNIQUE> \
  --champion <CURRENT_CHAMPION> \
  --cycle <NN> \
  --prereg-sha <sha> \
  --family-size <N>
```

`compare_techniques.py` is the only thing in this repository permitted to issue
a verdict. Everything else reads the verdict string it returns. A candidate that
scores higher but does not clear the corrected bar is `INCONCLUSIVE`, and is
reported that way.

Promotion to champion happens through `tools/ledger.py promote`, which is the
sole write path to the `champion` field in `registry.json`. Editing that field
by hand desynchronizes it from the chain that justifies it.

## Registering a technique that ran outside the loop

A run that reported test metrics without going through the loop cannot simply be
adopted; there is nothing to derive its result folder from and nothing tying its
decision rule to a design fixed in advance. Notebooks `08` and `10` are both in
this state.

Bringing one in requires:

1. Adding the sanitized probability-export cell to the notebook.
2. Writing the preregistration that the run should have had, acknowledging in
   it that the design is being recorded after the fact.
3. Rerunning, which costs the GPU time again.
4. Deriving the result folder from the exported probabilities.
5. Adjudicating with an explicit `--family-size` that counts the unlock the
   original run already spent.

Step 5 is the part that is easy to get wrong. The original run *did* read the
test split, so that unlock is spent whether or not it was recorded, and the
family size must reflect it.

## Related documents

| Document | Covers |
|---|---|
| `docs/EXPERIMENTS.md` | The protocol a single experiment must follow. |
| `docs/RESULTS_SCHEMA.md` | The files and fields a result folder must contain. |
| `docs/RESPONSIBLE_USE.md` | Claims discipline and deployment limitations. |
| `docs/RELEASE_CHECKLIST.md` | What to verify before a public update. |
| `research_loop/STATE.md` | Where the loop actually is right now. |
