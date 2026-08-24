# Hidden environment and platform separation

Status: historical design input. Completed decisions and remaining work are owned
by `docs/plans/digital-twin-v4-execution-plan.md`; this file is not a backlog authority.

## Scope

Refactor `fid_lab/simulation/twin` so a continuously evolving app-user world
and a continuously evolving recommendation platform interact only through
rendered slates and observable events.

## Findings

- V1 initialized platform `satisfaction`, fatigue, and intent features as noisy
  transforms of latent truth. Logistic regression therefore received oracle
  proxies and its launch result was invalid model evidence.
- Candidate serving and latent state shared one state module, so the dependency
  boundary relied on discipline rather than code structure.
- The hidden response model received the full candidate object, including
  unexposed candidates, rank scores, and platform history features.
- Future signup schedules were stored in platform state before signup occurred.

## Invariants

- `platform`, `serving`, and `training` cannot import environment or latent
  modules.
- Platform state contains event-derived estimates and counters, never hidden
  truth or future lifecycle state.
- The environment receives only `ServedSlate.exposed_item_ids`.
- The platform receives only `ObservableResponse` events.
- Changing latent state while freezing observable state cannot change recall,
  coarse/fine scores, eligibility, or the exposed slate.
- Only the factual mixed A/B world generates future training samples.

## Moves

1. Extract observable state to `twin/platform/state.py`.
2. Move lifecycle, traffic, response, and latent state under `twin/environment`.
3. Introduce the neutral exchange contracts in `twin/exchange.py`.
4. Replace oracle proxy names with explicit estimates and counters.
5. Hide future signup and retention inside the environment.
6. Split platform and environment random seeds for held-out world evaluation.
7. Quarantine the V1 GPU result as an engineering canary.
8. Reuse the existing W&D, DCNv2, and MMoE blocks behind the same ranker
   artifact and request-aware objective; defer DeepFM until sparse FIDs exist.
9. Bound training materialization with a deterministic probability-carrying
   reservoir and retain the full online ecosystem scale.
10. Add whole-step chronological offline metrics and exact-checkpoint replay
    across held-out hidden environments.

## Acceptance

- Focused twin tests pass, including dependency and latent-intervention gates.
- Architecture lint has zero errors.
- A/A remains exact and candidate microbatching preserves outcomes.
- V2 is run on multiple held-out environment seeds on RTX 4090 before any model
  is called launchable.
