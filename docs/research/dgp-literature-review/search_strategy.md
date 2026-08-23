# DGP literature search strategy

Research question: what data-generating-process architecture can credibly simulate an industrial short-video Feed and Local-service recommender without leaking a hand-authored scoring rule to the evaluated models?

The search covers 2010–2026 work on recommender simulators, learned user-response models, sequential and long-horizon recommendation, recommender ecosystems, slate off-policy evaluation, feedback loops, and LLM-powered user simulators.

## Terms

- Primary: recommender simulator; user simulator; data-generating process; world model; user-response model.
- Broader: interactive recommendation; reinforcement-learning environment; digital twin; ecosystem simulation.
- Narrower: slate competition; watch-time survival; session return; provider dynamics; multi-behavior response.
- Causal: logged bandit feedback; IPS/SNIPS; doubly robust; off-policy evaluation; counterfactual slate evaluation; support overlap.
- Frontier: generative agents; LLM user simulator; neural state-space model; autoregressive behavior model.

## Execution and evidence boundary

OpenAlex was queried in broad, recent, methodology-specific, and exact-seed phases. The final successful snapshot contains 1,500 candidate records before relevance filtering. Exact seed searches covered RecSim, RecSim NG, KuaiSim, SARDINE, T-RECS, Virtual-Taobao, RecoGym, RL4RS, Sim2Rec, slate OPE, and long-term engagement work.

Five rate-limited Semantic Scholar Graph API searches were attempted for citation expansion. All returned HTTP 429 despite bounded retries and 1.1-second spacing; no API key was available. This limitation is preserved in `raw/semantic-scholar-errors.json` rather than silently replacing Semantic Scholar metadata with search-engine results. OpenAlex and primary publisher/arXiv pages were used for the retained corpus.

Filtering required a recommender/ranking context plus simulator, sequential, feedback-loop, or causal-evaluation content. Exact seed papers were retained even when OpenAlex lacked an abstract. Records were deduplicated by DOI and normalized title. The resulting corpus contains 64 papers across seven methodology groups. Citation counts are discovery aids, not quality scores; recent low-citation work is retained separately.

Raw API payloads are stored under `raw/`; the OpenAlex snapshot is gzip-compressed to keep the public repository bounded. `corpus.json`, `classifications.json`, and `lit_review_summary.json` are machine-readable derivations. The collector is deterministic given the upstream APIs, but the upstream indexes and citation counts can change.
