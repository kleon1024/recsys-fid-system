# L-RECALL-EXT-001 — External Feed retrieval ladder

Change type: Main Feed retrieval model  
Decision: retain Popular; hold Co-visit Graph; reject Two-Tower and Multi-interest

All arms use the same 7,388-item catalog, random-exposure evaluation users,
Top-50 budget, fixed fine-rank artifact, and independent Feed world. Model seed
changes do not change the evaluation split.

| Route | Mean Recall@50 | Shadow stay result | Decision |
|---|---:|---|---|
| Popular | 1.068% | Control | Retain |
| Co-visit Graph | 1.068% | No final-slate delta | Hold |
| Two-Tower | 0.712% | Two nonpositive seeds, one positive | Reject |
| Multi-interest | 0.850% | Mean negative, no stable seed pass | Reject |

The first smoke produced loss near 98 because sampled `log q` correction was
applied before temperature scaling and was amplified 12.5 times. That run is
invalid. The corrected loss falls to about 8.5--8.6. A second invalid aggregate
changed evaluation users with model seed; the final run freezes evaluation seed
20260824 and varies only model initialization and training order.

The learned routes cover 30--42% of the catalog instead of Popular's 0.68%, but
coverage alone does not compensate for lower Recall or unstable downstream
stay. The result diagnoses retrieval representation and positive-sample bias;
it is not evidence that the fixed fine ranker or V4 Feed world failed.

This is random-exposure offline and paired shadow evidence, not a live A/B or LT.
