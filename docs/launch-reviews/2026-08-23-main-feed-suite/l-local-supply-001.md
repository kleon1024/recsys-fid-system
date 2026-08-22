# L-LOCAL-SUPPLY-001 — Personalized POI posting supply

Status: `pass_to_cluster_switchback`, not a viewer-level rollout.

## Change

On the POI posting page, rank candidate POIs with author/history context and use
the same recommendation to improve selection, submit, publish, and predicted
content quality. Newly published videos mutate a copied supply catalog and then
enter normal Fresh/Local recall before Feed distribution.

## Posting funnel

| Metric | Control | Treatment |
|---|---:|---:|
| Posting-page entry | 5.00% | 5.00% |
| POI selection | 1.67% | 2.67% |
| Submit | 0.67% | 1.33% |
| Publish | 0.67% | 1.00% |
| Published videos | 2 | 3 |
| Mean published quality | 0.500 | 0.636 |

Entry is unchanged, so the synthetic increment comes from ranking and downstream
funnel quality rather than injecting more posting-page traffic.

## Supply-to-distribution effect

| Metric | Paired-world relative effect |
|---|---:|
| Feed stay | +1.20% |
| LT | +1.56% |
| HLT | +0.83% |
| POI anchor clicks | +6.82% |
| Local Service Value | +9.17% |
| Long-term Feed Value | +0.81% |

## Experiment unit and decision

Publishing supply changes the catalog seen by other viewers, violating viewer
SUTVA. UID randomization is invalid. The next real experiment unit must be a
city-day switchback or author cluster with contamination monitoring.

Pass only to that cluster experiment: all paired-world Feed guardrails are
positive, but three published treatment videos are not enough for a rollout
claim or stable confidence interval.
