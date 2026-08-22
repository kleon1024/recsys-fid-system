# Security Policy

## Supported scope

The current repository is a local reference implementation and public RFP. It is not a production service and does not accept production data, credentials, or private infrastructure configuration.

Security review covers the current `main` branch.

## Reporting a vulnerability

Do not disclose a suspected vulnerability in a public issue or RFP response. Use GitHub Private Vulnerability Reporting for this repository. Include:

- affected commit and file;
- reproduction conditions;
- potential impact;
- whether credentials, personal data, or third parties may be affected;
- a minimal suggested mitigation if known.

Do not access data, accounts, or infrastructure that you do not own or have explicit permission to test. Do not include real secrets or personal data in a report.

## Public RFP boundary

RFP questions and capability statements must not include:

- credentials or production endpoints;
- customer, employee, or production event data;
- confidential employer or client material;
- unpublished security findings;
- private commercial pricing.

## Dependency and production expectations

The awarded production implementation must define supported versions, automated dependency review, software-bill-of-materials generation, container and infrastructure scanning, secret management, least privilege, incident response, and remediation service levels. These controls are RFP deliverables and are not implied by this local simulator.
