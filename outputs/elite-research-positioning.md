# Elite Research Positioning

## Revised Paper Direction

The upgraded paper should center **selective, conservative fairness control at deployment time**, not always-on universal RL.

## Why The Shift Is Better Supported

### 1. Post-processing is now a strong mainstream research direction

- FRAPPE shows that modular post-processing can inherit strong fairness-utility trade-offs from in-processing methods:
  - https://proceedings.mlr.press/v235/tifrea24a.html
- LinearPost argues that post-hoc fairness can be framed as a general optimization problem with strong guarantees in the high-fairness regime:
  - https://arxiv.org/abs/2405.04025

### 2. The community increasingly cares about fairness after deployment

- Runtime fairness monitoring makes the case that static offline fairness is insufficient:
  - https://arxiv.org/abs/2305.04699
- Fairness under distribution shift is now a recognized problem setting:
  - https://arxiv.org/abs/2402.01327
- Long-term fairness surveys emphasize dynamics, feedback loops, and operational deployment issues:
  - https://arxiv.org/abs/2406.06736

### 3. Selective intervention is a promising modern angle

- Selective debiasing at inference time is explicitly motivated as a safety mechanism when retraining is impractical:
  - https://arxiv.org/abs/2407.19345
- Minimal-change post-processing is now an explicit research objective:
  - https://arxiv.org/abs/2408.15096
- BiasGuard reinforces the framing of fairness mitigation as a production guardrail:
  - https://arxiv.org/abs/2501.04142

### 4. RL alone is still hard to justify as the main story

- The current fair-RL literature is still early and fragmented:
  - https://arxiv.org/abs/2405.06909
- This matches the current FairFlow evidence: always-on RL can recover fairness, but it over-intervenes and is order-sensitive.

## Practical Scientific Takeaway

The strongest credible claim is:

> A deployment-time fairness guard should be selective, conservative, and model-agnostic.

FairFlow’s RL component is still valuable, but it is much more defensible as an **anchored selector over candidate fairness corrections** than as an unrestricted always-on controller.

