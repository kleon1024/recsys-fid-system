# Recommendation evolution implementation plan

## Scope

Restructure public documentation, integrate the existing scale draft, and add
stage-specific samples, commerce attribution, model evolution benchmarks,
trained generative retrieval, ClickHouse diagnostics, and production failure
drills.

## Baseline findings

- Remote and local `main` are `c3ac52b`.
- All 28 committed tests pass.
- Five deterministic demos provide the semantic golden baseline; online latency
  is excluded because wall-clock timings are intentionally non-deterministic.
- The uncommitted `fid_lab/scale` draft has three orphan-module findings and is
  not accepted state.
- The Windows RTX 4090 is reachable with PyTorch 2.7.1 and CUDA available.

## Execution order

1. Move durable root documentation into architecture, operations, and interview
   sections; update links and the public-doc gate without changing behavior.
2. Integrate scale distribution and tensor contracts with direct tests.
3. Add recall, coarse-rank, and fine-rank example authorities plus closed-loop
   commerce and seven-day time-decayed Pixel attribution.
4. Add comparable recall, pre-rank, fine-rank, and generative model families.
5. Add one benchmark runner with frozen temporal splits, equal budgets, model
   cards, and JSON/Markdown output.
6. Add ClickHouse lineage and diagnostic queries plus failure runbooks.
7. Run focused tests, the complete local gate, the million-impression benchmark,
   and the ten-million-impression GPU profile.

## Invariants

- Existing public Python behavior remains unchanged during the documentation
  move.
- Unexposed candidates are never ordinary fine-rank negatives.
- A task is negative only after its event-time and allowed-lateness window
  closes; unobservable Pixel outcomes remain masked.
- Sampling probabilities and every served stage score remain attached to the
  example, while predictions never become labels.
- Recall models share query corpus and candidate budgets; ranking models share
  one frozen candidate set and temporal split.
- Synthetic results are never represented as company-internal evidence.

## Verification

Run `python3 -m fid_lab.check` as the single repository gate. It must include
architecture lint, tests, deterministic demos, model-evolution smoke checks,
SQL contract checks, and public-document validation.
