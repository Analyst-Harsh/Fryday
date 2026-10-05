# Security Policy

Fryday is an early-stage learning project maintained by a single developer. This policy is intentionally minimal rather than promising a formal process that doesn't exist.

## Reporting a vulnerability

**Please do not open a public GitHub issue for a security vulnerability.** Report it privately through [GitHub's private vulnerability reporting](https://github.com/Analyst-Harsh/Fryday/security/advisories/new) and include:

- a description of the vulnerability and its impact;
- steps to reproduce;
- any relevant logs, payloads or affected code paths.

There is no formal SLA and no bug bounty. Confirmed reports get a best-effort fix, with credit unless you'd rather not.

## Secrets policy

- **No secrets in git, ever.** Secrets live only in an untracked `.env` (git-ignored).
- `.env.example` holds placeholders only and documents every key the app reads.
- **gitleaks** scans staged changes in the lefthook pre-commit hook and the full history in CI. Never bypass it with `--no-verify`.
- A false positive gets an inline `gitleaks:allow` comment or a `.gitleaksignore` entry, with a reason, never a disabled hook.
- **If a secret is committed or pushed:** rotate (revoke) it first, since rotation is the only real fix. Then purge it from history (`git filter-repo`) and force-push. Assume anything pushed to this public repo has already been scraped.
