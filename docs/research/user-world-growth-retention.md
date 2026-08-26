# User-world growth, needs, and retention model

Status: implemented mechanics; calibration and standard-scale acceptance pending

This is a synthetic engineering reference. It does not claim to reproduce a
private production system or identify real acquisition and retention effects.

## State boundary

The user world is Markov or semi-Markov at the checkpoint boundary:

```text
hidden_state(t)
+ served_slate(t)
+ exogenous_counter_noise(t)
-> observable_events(t)
+ hidden_state(t+1)
```

It never retains the full event history. Its sufficient state contains bounded
interest and exposure memory, the current need episode, satisfaction, fatigue,
habit, activation, lifecycle stage, next-return time, acquisition context, and
pending delayed outcomes. Counter-based randomness makes the next transition
independent of request batching and replay order.

Historical events belong to the platform data authority. A world checkpoint
stores only the event-stream identity and committed cursor. The event store owns
immutable partitions, watermark, lateness, retention tiers, training examples,
and audit evidence independently of hidden-world evolution.

## Acquisition and product-led growth

Potential entrants are heterogeneous across organic, paid, referral,
creator-led, and cross-product channels. Each has a latent acquisition quality
and referral susceptibility. Paid acquisition responds to exogenous
country-level campaign intensity. Referral and creator-led acquisition respond
to bounded pressure created by factual share, follow, and publish events.

This creates interference deliberately:

```text
recommendation quality
-> satisfaction and sharing/publishing
-> PLG pressure
-> later registrations
-> future traffic and training data
```

An experiment that can affect this path must therefore declare whether it is a
short-term user-randomized test or a longer cluster/switchback ecosystem test.

## Heterogeneous need episodes

Users do not have one stationary content preference. Each carries a temporary
need kind, topic, strength, and expiry. Entertainment, information, social,
local, commerce, and creation needs alter surface-entry probability and response
without becoming observable model features. Expired needs resample from the
user's long-run interests plus exogenous variation. Positive outcomes partially
satisfy a need; disappointment and negative feedback preserve or intensify it.

## Activation and retention

Acquisition quality is not retention. Session outcomes update an activation
score and session-value EMA. Session starts update return streak and lifecycle
stage. Session end produces a delayed return interval and possible churn from
satisfaction, fatigue, habit, activation, recent value, acquisition quality,
need strength, and churn susceptibility.

The platform observes registrations, sessions, actions, and eventual returns.
It does not observe latent need strength, acquisition quality, activation truth,
or churn probability.

## Research basis

- [RecSim](https://research.google/pubs/recsim-a-configurable-simulation-platform-for-recommender-systems/) motivates configurable latent user state, preference dynamics, and sequential choice rather than static independent responses.
- [RecSim NG](https://research.google/pubs/recsim-ng-toward-principled-uncertainty-modeling-for-recommender-ecosystems/) motivates modular probabilistic multi-agent state and accelerated simulation, while requiring calibration to observed data for sim-to-real claims.
- [Reinforcing User Retention in a Billion Scale Short Video Recommender System](https://arxiv.org/abs/2302.01724) models short-video recommendation as a request-level process whose delayed return interval connects sessions and retention.
- [Towards Content Provider-Aware Recommendation Systems](https://research.google/pubs/towards-content-provider-aware-recommendation-systems-a-simulation-study-on-interplays-among-user-and-provider-utilities/) motivates a separate partially observable provider process coupled through recommendation exposure and feedback.
- [Feedback-loop simulation](https://arxiv.org/abs/2510.14857) shows why repeated retraining can change concentration and user homogenization, so diversity and segment effects must be audited across evolving worlds.

These sources justify mechanisms, not coefficients. Coefficients remain
synthetic until fitted against an explicit public or licensed trajectory source.

