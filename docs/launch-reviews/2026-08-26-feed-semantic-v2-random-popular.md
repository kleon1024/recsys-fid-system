# Feed semantic-v2 Random to Popular review

Evidence class: executable synthetic-world A/B

Source commit: `b994d5a`

The first semantic-v2 run removed the old 64-topic/item-ID cycle, replaced
contiguous Random windows with counter-random permutations, compared one route
per arm without RRF, and wrote 64 event-time Launch Bundle partitions.

The catalog audit observed all 512 topics and an adjacent-topic increment rate
of 0.231%, so the former deterministic cycle is absent. Popular matched the
user country for every candidate. Control and treatment contained 751 and 723
triggered users.

Naive engagement Popular did not pass. Long View increased 6.10%, but Stay fell
3.20% with a confidence interval crossing zero, while negative feedback rose
23.28% with a strictly positive 95% interval. The original gate incorrectly
reported this as inconclusive because it checked an underpowered primary before
the conclusive guardrail. The gate now rejects a significant guardrail breach.

Request evidence showed that Popular selected higher observed engagement and
public quality, but exposed a much smaller item pool. This is consistent with a
polarizing engagement objective: an engagement event count is not qualified
user value. The next adjacent strategy is Bayesian-qualified Popular using the
same decayed impression window and a decayed negative-feedback penalty. That is
a new policy version and must rerun against unbiased Random; the old result is
retained as a rejected launch, not overwritten.

No result in this document is a TikTok or production metric.
