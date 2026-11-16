# v0.1.0 — reproducible KKT-guided Qwen3 active-set repair

Initial research release with English/Chinese introductions, mathematical method, portable environment setup, and sequential reproduction commands.

Included: synthetic QP generation, independent KKT verification, numerical baselines, Qwen3-0.6B LoRA SFT, custom multi-turn GRPO with reward A/B/C, matched evaluation, frozen reference results and case studies.

Attachments provide all 7,000 QPs and teacher data, four final trained adapters, raw evaluation trajectories and training audits. Use the committed `docs/release_manifest.json` and `scripts/restore_artifacts.py` to verify/restore them. Base Qwen weights are downloaded separately at the pinned revision.

The reference GRPO-B result is 85.10% ID verified success versus 78.57% for SFT, with one training seed. Three inference seeds are not three independent training seeds. Numerical baselines remain strong; see the report for limitations.

Release validation is recorded in `docs/VALIDATION.md`. Training was not rerun during packaging. Historical raw-residual roundoff differences are logged without relaxing the KKT success criterion.
