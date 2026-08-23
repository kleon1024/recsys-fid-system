# Contributing

This repository is an executable recommendation-system reference and public
RFP. Contributions should improve a declared contract, reproducible experiment,
failure diagnostic, or public explanation.

## Development setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m fid_lab.check
```

The final command is the repository acceptance gate. Focused tests are useful
during development, but they do not replace it.

## Change contract

1. State the invariant and the subsystem that owns it.
2. Preserve point-in-time labels, sampling probabilities, artifact versions and
   the distinction between business Value Trees and the platform LT container.
3. Compare models on the same time split, corpus, candidate budget and seeds.
4. Add a failing regression test for a bug or a measurable benchmark for an
   algorithmic change.
5. Regenerate affected reports and update `reports/launches/MANIFEST.sha256`.
6. Record rollout, hold or rejection in a Launch Review; higher AUC alone is not
   launch evidence.

## Public evidence boundary

- Use synthetic, licensed or explicitly consented data only.
- Do not submit employer code, internal metrics, protected interview posts,
  credentials, endpoints, personal data or confidential documents.
- Distinguish public-source evidence, synthetic measurements and design
  assumptions.
- Do not claim that this repository reproduces a company's proprietary system.

## Pull requests

Keep one coherent change per pull request. Include the commands run, the
affected report hashes, known limitations and whether GPU evidence was produced.
Security issues must follow [SECURITY.md](SECURITY.md), not a public issue.
