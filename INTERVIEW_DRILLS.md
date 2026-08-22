# Interview Drills for the FID Lab

Use the running code as the reference system. For each drill, first state the contract and invariant, then derive or code the answer.

## 1. FID coding drill

Implement `pack(slot, signature)`, `unpack(fid)`, and V1-to-V2 conversion without reading `fid_lab/fid.py`.

Required cases:

- V1 slot 1023 succeeds; 1024 fails.
- V2 slot 32767 succeeds; 32768 fails.
- Signatures larger than their bit width are masked.
- V2 rejects a set reserved bit.
- Explain why conversion from V1 to V2 can collide.

Strong answer: separates the packed identifier from the function that creates the signature, uses unsigned-bit reasoning, and identifies the lossy six-bit truncation.

## 2. Stable feature hashing

Replace BLAKE2 with another deterministic hash and prove that the same raw event receives identical FIDs in two Python processes.

Follow-ups:

- Why is Python `hash()` unsafe here?
- Does increasing the bucket count require retraining?
- How do you estimate and monitor collision rate per slot?
- When should a raw ID remain collisionless rather than hashed?

Strong answer: versions the hash contract, salt/namespace, bucket count, and schema together; it does not silently change any of them at serving time.

## 3. Feature-combination framework

Add the cross `(country, category, device)` with a new slot. Ensure that `("ab", "c")` cannot serialize like `("a", "bc")`.

Follow-ups:

- Who owns the slot allocation?
- How do you prevent duplicate crosses implemented by two teams?
- How would you deprecate a cross without breaking old models?
- Should crosses be materialized in logs, computed in training, or computed in a feature service?

Strong answer: names one schema authority, uses an unambiguous serialization, versions it, and requires training-serving parity tests.

## 4. Hand derivation: FM and DeepFM

Given field embeddings `v_i`, derive the second-order FM interaction:

```text
sum_{i<j} <v_i, v_j>
= 1/2 * sum_k ((sum_i v_i,k)^2 - sum_i v_i,k^2)
```

Explain why the right side changes complexity from pairwise field comparisons to linear work in the number of fields. Then derive the gradient with respect to one embedding `v_i`.

DeepFM's final logit in this lab is:

```text
z = first_order(x) + fm_second_order(x) + dnn(concat(embeddings))
p(click | x) = sigmoid(z)
```

Follow-up: explain precisely how FFM differs. In FFM, feature `i` uses a different embedding when interacting with field `j`, so `v_i` becomes `v_i,j`.

## 5. AUC and log loss

For labels `[0, 1, 0, 1]` and predictions `[0.1, 0.8, 0.7, 0.6]`:

- Compute AUC as the fraction of positive-negative pairs ranked correctly, including the tie rule.
- Compute binary log loss.
- Explain why a monotonic transformation can preserve AUC while damaging calibration and log loss.
- Explain why global AUC can improve while per-user ranking quality gets worse.

Strong answer: treats AUC as a ranking probability, distinguishes calibration, and proposes sliced metrics plus NDCG or group AUC when the product ranks within users.

## 6. Chain consistency system design

Design the path from event logging to online scoring:

```text
event -> schema/version -> feature transform -> FID -> training example
      -> model artifact -> deployment -> online feature transform -> score
```

Required invariants:

- Same slot registry, signature/hash version, cross serialization, defaults, and bucket sizes offline and online.
- Event-time joins do not read future information.
- A model declares the exact feature-schema version it accepts.
- Rollback restores both model and compatible feature configuration.
- Shadow traffic compares FIDs and scores before cutover.

Diagnose these failures:

1. Offline AUC is stable but online CTR falls immediately after a hash-library upgrade.
2. Only one country regresses after adding a feature default.
3. V1 and V2 models share an embedding service during migration.
4. A new ANN index is built from item tower version N while traffic uses user tower version N+1.

## 7. Intensive project examination

An interviewer asking for an "intensive project examination" is testing whether you owned the causal system, not whether you can recite the architecture. Use this lab to rehearse:

- Why was a packed FID needed instead of `(feature_name, value)` strings?
- What was the exact training-serving invariant?
- Which collision or migration failure did the tests prevent?
- Why did the initial DeepFM smoke have poor log loss?
- What evidence justified moving from XGBoost to embeddings?
- What would have to be true before adding DIN, MMoE, PLE, or a Transformer?

A strong project answer contains an observed problem, scale and constraints, a technical decision, rejected alternatives, measurable offline and online results, failure analysis, and your exact ownership.

## 8. Practical extension order

Do not add architectures merely because they are newer. Extend the lab in this order:

1. Add point-in-time-correct event histories, then implement DIN or a sequence Transformer.
2. Add multiple labels such as click, long-view, and conversion, then implement MMoE or PLE.
3. Split retrieval from ranking, create an ANN item index, and enforce user-tower/item-index version consistency.
4. Add delayed-label handling, calibration, bias correction, and online experiment guardrails.

Each extension must keep the existing FID/schema authority and add a test for its new consistency boundary.
