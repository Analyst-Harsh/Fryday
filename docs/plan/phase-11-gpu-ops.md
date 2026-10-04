# Phase 11 — GPU ops tooling

| | |
|---|---|
| **Goal** | Choose a GPU provider and make GPU sessions safe, scripted and cheap. |
| **Depends on** | 9 |
| **Estimate** | 3–5 working days at 2–4 h/day |
| **HLD refs** | §6, §8 |
| **Status** | Not started |

## Scope
- Choose the provider from a price comparison and these criteria: direct VM SSH with `-L` forwarding, Docker + NVIDIA toolkit, `SYS_ADMIN` allowed (Nsight), a 24 GB card, and persistent model cache.
- Provider-agnostic `gpu-up` / `gpu-down` scripts with secrets injection and a disk wipe.
- Hardened SSH tunnel (HLD §6): a per-session key, key-only sshd, `PermitOpen`, services bound to 127.0.0.1, an autossh sidecar, keepalives, and `caffeinate`.
- Dead-man switch and a GPU heartbeat alert.
- GPU runbook entries: box dead, tunnel down.
- One short session to run S0-11 (tunnel with gRPC streaming + HTTP + reconnect), S0-6 (Nsight counters) and S0-7 (24 GB fit).

## Out of scope
- Training and serving work.

## Exit criteria
- [ ] The provider choice and its criteria are recorded.
- [ ] `gpu-up` and `gpu-down` work.
- [ ] A test shows the dead-man switch kills the box.
- [ ] S0-6, S0-7 and S0-11 results are recorded.
- [ ] The session cost is recorded.

## What you learn
Infrastructure automation, SSH hardening, GPU instance basics (drivers, nvidia-smi, containers).

## Units
_Written when this phase starts. Each unit is 2–3 h and follows the `superpowers:writing-plans` task format (files, failing test, implementation, verification, commit)._
