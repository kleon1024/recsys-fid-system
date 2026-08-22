# Bidder Response Template

Use this structure exactly. Public responses must exclude confidential pricing, credentials, customer data, and protected implementation details.

## 1. Bidder identity

- Legal entity:
- Operating country or countries:
- Primary contact:
- Public website:
- Proposed lead engineer:
- Proposed delivery team:
- Earliest start date:

## 2. Executive response

In no more than 500 words, state:

- your understanding of the required outcome;
- the largest technical and delivery risks;
- your proposed architecture direction;
- why your team can own the full causal chain;
- material exceptions to the RFP.

## 3. Relevant evidence

For each comparable project:

| Field | Response |
|---|---|
| Product and use case | |
| Traffic and data scale | |
| Your exact ownership | |
| Data and training stack | |
| Retrieval and serving stack | |
| Availability and latency | |
| Measured outcome | |
| Failure or incident handled | |
| Verifiable reference | |

Redact customer identity where required, but preserve enough evidence to evaluate scale, ownership, and acceptance.

## 4. Proposed architecture

Provide:

- system context and deployment diagram;
- event and data lineage;
- feature and FID ownership;
- training and Parameter Server design;
- retrieval, ranking, policy, and mixing design;
- artifact publication and rollback unit;
- offline/online consistency method;
- observability and incident isolation;
- security and privacy controls;
- dependency and build-versus-buy decisions.

For each major component, state the invariant, owner, fallback, measurable acceptance, and rejected alternative.

## 5. Scale tiers

Complete all tiers:

| Metric | Tier A: 100 RPS | Tier B: 1,000 RPS | Tier C: 10,000 RPS |
|---|---:|---:|---:|
| Assumed item count | | | |
| Assumed active users | | | |
| Event throughput | | | |
| Recall candidates | | | |
| Fine-rank candidates | | | |
| p50 latency | | | |
| p95 latency | | | |
| p99 latency | | | |
| Availability | | | |
| Update freshness | | | |
| Monthly infrastructure cost | | | |
| Main bottleneck | | | |

Explain the benchmark method and every assumption.

## 6. Delivery plan

For each gate G0-G6:

| Gate | Duration | Team | Deliverables | Acceptance command/evidence | Owner decision |
|---|---:|---|---|---|---|
| G0 | | | | | |
| G1 | | | | | |
| G2 | | | | | |
| G3 | | | | | |
| G4 | | | | | |
| G5 | | | | | |
| G6 | | | | | |

Identify the critical path, external dependencies, and assumptions.

## 7. Data and model methodology

Describe:

- impression/action identity and delayed-label joining;
- negative sampling and propensity treatment;
- point-in-time features and temporal validation;
- baseline and candidate models;
- multi-task architecture and value fusion;
- calibration and slice evaluation;
- online or nearline update design;
- model/index compatibility and rollback;
- A/B design and guardrails.

## 8. Generative recommendation option

If proposed, describe and price separately:

- Semantic-ID construction and lifecycle;
- valid constrained decoding;
- new-item and streaming updates;
- multi-objective generation;
- ANN baseline and equal-budget benchmark;
- latency, compute, rollback, and operational risks.

Write `Not proposed` if this is outside your recommended first delivery.

## 9. Security, privacy, and reliability

Provide:

- threat-model method;
- identity, secret, network, encryption, and environment controls;
- dependency and supply-chain controls;
- retention and deletion design;
- SLO and alerting proposal;
- backup, restore, failover, and disaster-recovery design;
- proposed fault-injection and recovery tests;
- unresolved risk acceptance process.

## 10. Team and availability

| Person | Role | Allocation | Relevant ownership | Location/timezone | Engagement duration |
|---|---|---:|---|---|---|
| | | | | | |

Name all subcontractors and the work they would perform.

## 11. Commercial response

Submit this section privately.

| Item | Fixed price | T&M estimate | Third-party cost | Assumptions |
|---|---:|---:|---:|---|
| G0 | | | | |
| G1 | | | | |
| G2 | | | | |
| G3 | | | | |
| G4 | | | | |
| G5 | | | | |
| G6 | | | | |
| Optional generative track | | | | |
| Warranty/support | | | | |

State currency, tax treatment, payment terms, rates, change-request terms, and proposal validity period.

## 12. IP, licenses, and dependencies

List:

- bidder background IP;
- proposed open-source dependencies and licenses;
- managed services and lock-in implications;
- external models, datasets, APIs, and usage terms;
- artifacts that cannot transfer to the owner;
- exceptions to the RFP ownership requirement.

## 13. Assumptions and exceptions

| ID | RFP section | Assumption or exception | Impact | Required owner decision |
|---|---|---|---|---|
| | | | | |

Silence means acceptance of the RFP requirement.

## 14. References

Provide two references or verifiable public artifacts demonstrating comparable ownership. State whether the owner may contact each reference during evaluation.

## 15. Declaration

Confirm that:

- the response is accurate;
- the proposed team is available as stated;
- no confidential third-party material was used;
- all material assumptions and exceptions are disclosed;
- pricing and evidence remain valid for the stated proposal period.

Authorized representative:

Date:
