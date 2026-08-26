# Feed publishing world closure review

Date: 2026-08-26  
Decision: world/sample gate passed; Publish Queue model not launched  
Evidence scope: synthetic-world causal evidence only

## Intervention and invariant

The intervention remained a viewer-UID Feed retrieval A/B: control used Random
and treatment added Popular through RRF. Both arms used the same accepted fine
ranker and value policy. Feed exposure could change hidden creator inspiration,
which could later produce posting entry, create, publish, immutable post
allocation, lifecycle ingestion and future Feed eligibility. The platform could
observe only events and point-in-time features; it could not read inspiration.

## Allocator failure and correction

The first run produced 2,836 create attempts, 745 successful publications and
1,088 failures. Of the failures, 980 were `NO_CAPACITY`. The allocator had
incorrectly required a dormant immutable post slot to be preassigned to the
same creator even though publication binds the actual creator. The corrected
allocator draws globally from never-used slots and still prevents ID reuse.

The identical rerun produced:

| Event | Count |
|---|---:|
| Create | 2,967 |
| Publish success | 1,634 |
| Publish failed | 240 |
| Unique creating users | 1,384 |
| Unique publishing users | 1,070 |

All remaining failures were creator cooldown. `NO_CAPACITY` fell to zero.

## Feed A/B evidence

The fixed run used 10,000 users, 100,000 content slots, 112 burn-in ticks and 64
experiment ticks on the RTX 4090. There were 3,175 control and 3,094 treatment
triggered users.

| Metric per triggered user | Relative delta | 95% interval for absolute delta |
|---|---:|---:|
| Stay | +14.57% | +8.88s to +19.02s |
| 3-second play | +11.36% | +0.87 to +2.42 |
| Long view | +22.28% | +1.05 to +1.81 |
| Complete | +25.62% | +0.64 to +1.07 |
| Negative feedback | -5.77% | -0.145 to -0.003 |

Popular was promoted over Random for the declared synthetic Feed metrics. This
review does not attribute publication lift to a Publish Queue model because no
such model was served in this experiment.

## Sample-system correction

Feed publishing value now has its own cross-request `PublishQueueExample`
authority. Candidates remain Feed content. Labels are posting entry and create
within 24 hours and publish within 48 hours, with maturity masks, factual
propensity and globally unique observable engaged-last-touch attribution. The training materializer
retains later posting requests by the same viewer instead of filtering the log
to the Feed request ID. Hidden inspiration remains unavailable to the Joiner.

The next Launch Review must train and shadow an independent Publish Queue
scorer, add qualified-post quality after its maturity window, and change only
its calibrated contribution in the common Feed mixer.
