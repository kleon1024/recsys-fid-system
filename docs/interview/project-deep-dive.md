# Senior recommendation project deep dive

## Two-minute answer

I built a public production-like POI recommendation reconstruction covering
both creator supply and viewer distribution. The main Feed path has multi-route
recall, coarse rank, multi-task fine rank, value fusion, constraints, and
cross-business mixing; POI posting, map/detail, YMAL, product, and review remain
separate model surfaces. The data path starts from authoritative decisions and
exposures, closes task-specific labels in event time, separates closed-loop
payment from open-loop Pixel attribution, and emits different samples for
recall, coarse rank, and fine rank. I compared LR, XGBoost, WDL, DeepFM,
DCN-Mix, DIN, MMoE, and PLE through mature open-source implementations on the
same temporal split and candidate budget. The important result was not that a
larger model always won: a linear DGP favored LR, while cross features made
XGBoost and distilled DCN stronger, which is why production model iteration
must connect architecture to an observed failure mode.

## What did the algorithm change?

The algorithm changes opportunity and ordering, not merely page entry. Recall
adds relevant long-tail and intent candidates; coarse rank protects candidates
the heavy model would value; fine rank predicts separate consumption, POI,
transaction, and negative-feedback outcomes; the value layer chooses the
trade-off. Product placement or location authorization changes eligibility, but
it does not replace learned relevance or value estimation.

## Why did AUC improve but A/B not move?

I first verify assignment, trigger, mature labels, and offline-online replay.
Then I locate the intervention: a recall change needs Recall@K, a coarse change
needs fine-winner preservation, and a value change needs calibration and slate
metrics. Global AUC is conditional on the evaluated candidates and can rise
while recall opportunity, top-K order, calibration, trigger rate, or experiment
power remains unchanged. The simulator injects known product, model, and
strategy effects and checks whether A/B intervals recover the true ITT.

## Why three sample tables?

Recall learns corpus discrimination and needs sampled negatives. Coarse rank
learns the real recalled distribution and teacher order. Fine rank learns only
from exposed items because skipped unexposed candidates have no observable
outcome. Combining them would turn retrieval assumptions into ranking labels.

Feed publishing value is separate again: its candidate is still Feed content,
but its label occurs on a later posting request. I use a dedicated Publish
Queue sample with 24/48-hour maturity and observable multi-touch attribution,
then validate incremental creator response through viewer-UID A/B. This must
not be confused with ranking POI, music, or product candidates on the posting
page.

## What was the hardest failure?

The hardest class is false negatives from delayed or unobservable conversion.
Closed-loop payment can be joined through transaction IDs after its window
closes. Open-loop Pixel data can be duplicated, blocked, late, or missing an
identity, so absence of a callback is not automatically a zero label. The
Joiner uses task masks and normalized time-decayed fractional attribution,
while monitoring match coverage separately from conversion rate.
