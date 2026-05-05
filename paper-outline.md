# Paper Outline

> Note: this outline currently reflects an older result bundle.  
> The next revision should use the newer framing:
> `group_threshold` and `rule_based` as **offline calibrated static baselines**,
> versus `guard_threshold`, `primal_dual_guard`, and related methods as
> **online adaptive deployment controllers**.

## Framing Decision

`group_threshold` should remain a baseline.

It is not scientifically honest to claim plain group-threshold post-processing as our own method, because it is a standard post-processing family already represented in the fairness literature.

What we **can** claim as our method is:

- **FairFlow-Guard**: a selective threshold guard that activates a strong fairness post-processor only when rolling fairness degrades
- **FairFlow-Guard-RL**: an optional adaptive variant where RL acts as a conservative selector over candidate corrections

So the paper should be framed as:

> We do not introduce group-threshold post-processing itself. We introduce a deployment-time control layer that decides **when** and **for how long** fairness correction should be activated.

## Working Title

**FairFlow-Guard: Selective Threshold Guardrails for Post-Deployment Fairness Control**

Alternative title:

**Selective Fairness Guardrails for Black-Box Classifiers in Dynamic Deployment Settings**

## One-Sentence Thesis

Always-on fairness correction is effective but operationally costly, while always-on RL is unstable under stream shift. A selective threshold guard offers a stronger deployment trade-off by activating fairness correction only when rolling fairness deteriorates.

## Core Problem Statement

Given a deployed black-box binary classifier, can we improve group fairness at inference time without retraining the base model, while minimizing unnecessary interventions on already-fair streams and remaining robust under changing stream order?

## Final Method Positioning

### Baselines

- `base_model`
- `group_threshold`
- `rule_based`
- `universal_rl`

### Proposed Main Method

- `guard_threshold`, to be renamed in the paper as **FairFlow-Guard**

### Secondary Variant / Ablation

- `fairflow_guard_rl`, to be renamed in the paper as **FairFlow-Guard-RL**

## Main Claims To Support

1. Static fairness post-processing is strong, but always-on correction spends too many interventions on already-fair streams.
2. A selective threshold guard recovers much of the fairness benefit of always-on post-processing while reducing intervention and equalized-odds distortion.
3. Always-on universal RL remains brittle under adversarial stream order, making it a weaker deployment choice than selective guarding.
4. RL-based selective acceptance is an interesting extension, but in the current evidence it is not stronger than the simpler threshold guard.

## Target Contribution List

1. A deployment-time formulation of fairness mitigation as **selective guardrailing**, not unconditional post-processing.
2. A conservative controller, **FairFlow-Guard**, that combines:
   - rolling fairness monitoring,
   - fairness-triggered activation,
   - hysteresis and cooldown,
   - minimal-change threshold-based intervention.
3. A multi-seed benchmark across 5 tabular datasets, 2 model families, and 6 mitigation strategies.
4. Ablations isolating the effect of selectivity, hysteresis, and RL anchoring.
5. A robustness study showing that selective threshold guards are much more stable than always-on RL under order shift.

## Section Structure

### 1. Introduction

#### 1.1 Motivation

- High-stakes classifiers are often already deployed.
- Retraining is not always feasible due to ownership, governance, or latency constraints.
- Post-processing is attractive, but most fairness work assumes a static evaluation setting rather than a monitored deployment setting.

#### 1.2 Deployment Gap

- Fairness is not a one-time property.
- The same model can look acceptable offline and become problematic along a stream.
- A deployment layer should be selective, conservative, and low-touch.

#### 1.3 Research Question

- Can selective fairness activation outperform always-on fairness correction as a deployment strategy?

#### 1.4 Contributions

- Introduce FairFlow-Guard as a selective deployment-time fairness controller.
- Show that always-on group-threshold correction is strong but intervention-heavy.
- Show that selective threshold guarding provides a better accuracy-fairness-intervention balance.
- Show that always-on universal RL is highly order-sensitive.

### 2. Related Work

#### 2.1 Post-Processing Fairness

- Hardt-style post-processing lineage.
- Modular and unified post-processing frameworks.
- Post-processing with minimal changes.

#### 2.2 Fairness Under Shift and Dynamic Environments

- Distribution-shift fairness surveys.
- Runtime fairness monitoring.
- Long-term fairness and operational fairness concerns.

#### 2.3 RL-Based Fairness Methods

- Fair RL remains promising but immature for deployment-time control.
- Existing RL fairness approaches do not directly solve low-touch selective deployment.

#### 2.4 Selective and Inference-Time Debiasing

- Selective debiasing and guardrail-style intervention.
- Conservative correction and intervention budgeting.

#### 2.5 Gap Summary

- The key gap is not "post-processing exists," but rather:
  - how to deploy fairness correction selectively
  - how to avoid unnecessary intervention
  - how to remain stable under stream dynamics

### 3. Problem Formulation

#### 3.1 Setting

- Binary classification stream
- Base model emits score `s_t` and base decision `\hat{y}_t`
- Protected attribute `a_t` is available to the post-processor
- True label `y_t` is used for offline evaluation

#### 3.2 Objective

- Improve demographic parity ratio
- Keep accuracy high
- Reduce equalized-odds distortion
- Minimize intervention rate
- Avoid fairness overshoot

#### 3.3 Deployment Constraints

- Black-box compatibility
- No base-model retraining
- Low overhead
- Safe behavior when fairness is already acceptable

### 4. Method

#### 4.1 Base Components

- Base classifier
- Static group-threshold post-processor
- Rolling fairness monitor

#### 4.2 Why `group_threshold` Is a Baseline, Not the Contribution

- The thresholding rule itself is not novel.
- The novel part is the online controller that governs when threshold corrections should be activated.
- This distinction should be made explicitly in the paper to avoid reviewer pushback.

#### 4.3 FairFlow-Guard

- Warm-up period before any intervention
- Activation when rolling DPR falls below a trigger
- Release when fairness returns to a safer band
- Hysteresis and cooldown to prevent oscillation
- Candidate corrections inherited from the threshold baseline

#### 4.4 FairFlow-Guard-RL

- RL does not directly control all outputs
- RL only accepts or rejects candidate corrections while the guard is active
- This is an exploratory adaptive extension, not the main method claim

#### 4.5 Main Design Hypothesis

- Threshold-based correction is strong
- The deployment challenge is deciding **when to turn it on**
- Therefore, selectivity is the central algorithmic contribution

### 5. Experimental Design

#### 5.1 Datasets

- Adult
- German Credit
- COMPAS
- Bank Marketing
- Recruitment

#### 5.2 Base Models

- Logistic Regression
- XGBoost

#### 5.3 Compared Methods

- `base_model`
- `group_threshold`
- `rule_based`
- `universal_rl`
- `guard_threshold` = **FairFlow-Guard**
- `fairflow_guard_rl` = **FairFlow-Guard-RL**

#### 5.4 Metrics

- Accuracy
- Demographic parity ratio
- Equalized odds gap
- Intervention rate
- Rolling tail average DPR
- Rolling tail fairness rate

#### 5.5 Statistical Protocol

- 3 seeds: `42`, `52`, `62`
- 5 datasets
- 2 base model families
- mean and 95% confidence intervals
- paired win counts against `group_threshold`

### 6. Main Results

#### 6.1 Overall Benchmark

Current overall method means from `elite_benchmark/method_summary.csv`:

- `base_model`: accuracy `0.8140`, DPR `0.7707`, EO gap `0.1060`, intervention `0.0000`, tail fair rate `0.4751`
- `group_threshold`: accuracy `0.8048`, DPR `0.9576`, EO gap `0.1037`, intervention `0.0688`, tail fair rate `0.6905`
- `guard_threshold` / FairFlow-Guard: accuracy `0.8077`, DPR `0.8920`, EO gap `0.0875`, intervention `0.0422`, tail fair rate `0.6129`
- `rule_based`: accuracy `0.8088`, DPR `0.8479`, EO gap `0.0727`, intervention `0.0205`, tail fair rate `0.5810`
- `fairflow_guard_rl`: accuracy `0.8113`, DPR `0.8288`, EO gap `0.0792`, intervention `0.0136`, tail fair rate `0.5442`
- `universal_rl`: accuracy `0.7954`, DPR `0.9930`, EO gap `0.0818`, intervention `0.0461`, tail fair rate `0.7689`

Interpretation:

- `group_threshold` is the strongest fairness baseline.
- `guard_threshold` preserves more utility than `group_threshold` while using fewer interventions and reducing EO gap.
- `fairflow_guard_rl` is too conservative to be the main method.
- `universal_rl` still looks strong on average fairness, but this is misleading without robustness analysis.

#### 6.2 Fairness-Pass Rates

Current fairness-pass rates:

- `group_threshold`: `93.3%`
- `universal_rl`: `86.7%`
- `rule_based`: `80.0%`
- `guard_threshold`: `76.7%`
- `fairflow_guard_rl`: `60.0%`
- `base_model`: `60.0%`

Interpretation:

- The main strength of `group_threshold` is undeniable and should be acknowledged.
- The main claim for FairFlow-Guard is not "best fairness at all costs," but "better deployment trade-off."

#### 6.3 Core Unfair Datasets

Average over `adult` and `compas`:

- `base_model`: accuracy `0.7743`, DPR `0.4577`, EO gap `0.1622`, intervention `0.0000`, tail fair rate `0.1256`
- `group_threshold`: accuracy `0.7574`, DPR `0.8903`, EO gap `0.1668`, intervention `0.1000`, tail fair rate `0.6093`
- `guard_threshold`: accuracy `0.7616`, DPR `0.7545`, EO gap `0.1392`, intervention `0.0714`, tail fair rate `0.4507`
- `rule_based`: accuracy `0.7615`, DPR `0.6392`, EO gap `0.0889`, intervention `0.0483`, tail fair rate `0.3601`
- `fairflow_guard_rl`: accuracy `0.7681`, DPR `0.5866`, EO gap `0.1091`, intervention `0.0300`, tail fair rate `0.2648`
- `universal_rl`: accuracy `0.7389`, DPR `0.8597`, EO gap `0.0814`, intervention `0.0894`, tail fair rate `0.6392`

Interpretation:

- `group_threshold` repairs fairness the most strongly.
- FairFlow-Guard recovers a substantial fraction of that fairness benefit with fewer interventions and lower EO distortion.
- This is the core section where the selective-guard contribution should be defended.

#### 6.4 Already-Fair / Low-Touch Datasets

Average over `german_credit`, `bank_marketing`, and `recruitment`:

- `base_model`: accuracy `0.8405`, DPR `0.9793`, EO gap `0.0685`, intervention `0.0000`, tail fair rate `0.7081`
- `group_threshold`: accuracy `0.8364`, DPR `1.0024`, EO gap `0.0616`, intervention `0.0480`, tail fair rate `0.7445`
- `guard_threshold`: accuracy `0.8384`, DPR `0.9837`, EO gap `0.0530`, intervention `0.0227`, tail fair rate `0.7210`
- `rule_based`: accuracy `0.8404`, DPR `0.9871`, EO gap `0.0619`, intervention `0.0020`, tail fair rate `0.7282`
- `fairflow_guard_rl`: accuracy `0.8402`, DPR `0.9902`, EO gap `0.0592`, intervention `0.0026`, tail fair rate `0.7305`
- `universal_rl`: accuracy `0.8330`, DPR `1.0818`, EO gap `0.0821`, intervention `0.0173`, tail fair rate `0.8553`

Interpretation:

- On already-fair datasets, selective methods behave much better than always-on RL.
- This supports the deployment motivation: do not spend interventions when the stream is already acceptable.

#### 6.5 Paired Wins Against `group_threshold`

FairFlow-Guard (`guard_threshold`) versus `group_threshold`:

- Accuracy: `22` wins, `5` losses, `3` ties
- EO gap: `18` wins, `9` losses, `3` ties
- DPR: `9` wins, `19` losses, `2` ties

Interpretation:

- FairFlow-Guard usually wins on utility and error-rate parity.
- `group_threshold` usually wins on demographic parity ratio.
- This is exactly the trade-off the paper should emphasize.

### 7. Ablation Studies

#### 7.1 Effect of Selectivity

Compare `group_threshold` vs `guard_threshold`.

Adult:

- `group_threshold`: accuracy `0.8565`, DPR `0.8032`, EO gap `0.2622`, intervention `0.0668`
- `guard_threshold`: accuracy `0.8609`, DPR `0.6877`, EO gap `0.2024`, intervention `0.0546`

COMPAS:

- `group_threshold`: accuracy `0.6947`, DPR `0.9397`, EO gap `0.0534`, intervention `0.0991`
- `guard_threshold`: accuracy `0.6990`, DPR `0.8151`, EO gap `0.0703`, intervention `0.0631`

Recruitment:

- `group_threshold`: accuracy `0.9894`, DPR `0.8833`, EO gap `0.0065`, intervention `0.0042`
- `guard_threshold`: accuracy `0.9893`, DPR `0.8907`, EO gap `0.0081`, intervention `0.0029`

Interpretation:

- Selectivity saves interventions consistently.
- On hard fairness datasets, selectivity sacrifices some DPR in exchange for better utility.
- On already-fair datasets, selective guarding is clearly preferable.

#### 7.2 Effect of Hysteresis

Compare `guard_threshold` vs `guard_threshold_no_hysteresis`.

- Adult: hysteresis is slightly better on both accuracy and DPR
- COMPAS: hysteresis improves DPR and tail fairness rate
- Recruitment: differences are tiny, which is expected in a near-fair dataset

Interpretation:

- Hysteresis is not a cosmetic feature; it stabilizes selective activation.

#### 7.3 Effect of Anchoring RL

Compare `universal_rl`, `guard_rl_no_anchor`, and `fairflow_guard_rl`.

Current finding:

- RL anchoring does improve stability relative to unrestricted RL
- but the RL variants still do not outperform the simpler threshold guard

Interpretation:

- RL should be presented as an adaptive extension, not the central contribution

### 8. Robustness Under Stream Reordering

#### 8.1 Main Robustness Result

This section is one of the strongest parts of the current paper.

Average across the order-stress benchmark:

`universal_rl`

- natural: accuracy `0.8274`, DPR `0.9163`, EO gap `0.0737`, intervention `0.0611`
- privileged-first: accuracy `0.7242`, DPR `1.8955`, EO gap `0.4159`, intervention `0.2030`
- unprivileged-first: accuracy `0.7163`, DPR `1.9424`, EO gap `0.4273`, intervention `0.2128`

`guard_threshold`

- natural: accuracy `0.8498`, DPR `0.7979`, EO gap `0.0936`, intervention `0.0402`
- privileged-first: accuracy `0.8498`, DPR `0.7846`, EO gap `0.1144`, intervention `0.0338`
- unprivileged-first: accuracy `0.8502`, DPR `0.7852`, EO gap `0.1150`, intervention `0.0341`

`fairflow_guard_rl`

- natural: accuracy `0.8519`, DPR `0.7017`, EO gap `0.0829`, intervention `0.0157`
- privileged-first: accuracy `0.8514`, DPR `0.7219`, EO gap `0.1084`, intervention `0.0138`
- unprivileged-first: accuracy `0.8517`, DPR `0.7200`, EO gap `0.1079`, intervention `0.0136`

Interpretation:

- Always-on RL is catastrophically unstable under adversarial order.
- FairFlow-Guard remains stable in both accuracy and intervention budget.
- FairFlow-Guard-RL is also stable, but it under-corrects fairness too much.

#### 8.2 Honest Limitation

- FairFlow-Guard is robust, but not fairness-optimal.
- Universal RL is fairness-aggressive, but not deployment-safe.
- This is a clean and honest scientific takeaway.

### 9. Discussion

#### 9.1 What The Results Mean

- The strongest static fairness baseline is still very hard to beat.
- The real contribution is deciding when to apply that baseline.
- Selective threshold guarding is the best current story in this project.

#### 9.2 Scientific Positioning

- We are not claiming a new fairness thresholding rule.
- We are claiming a new selective deployment mechanism for fairness correction.
- That claim is both more honest and better supported by the evidence.

#### 9.3 How To Write The Main Result

Recommended sentence:

> FairFlow-Guard recovers a substantial portion of the fairness gains of always-on group-threshold post-processing while reducing interventions, improving equalized-odds behavior, and dramatically improving robustness relative to always-on RL.

### 10. Limitations

#### 10.1 Fairness Notion

- The current study is primarily centered on demographic parity ratio, with EO gap used as a secondary fairness-quality metric.

#### 10.2 Label Availability

- The experiments are offline evaluations of a deployment-time controller.
- Real-world delayed-label settings remain future work.

#### 10.3 Protected Attribute Assumption

- The current setup assumes protected-group membership is available at inference time.

#### 10.4 Remaining Weakness

- FairFlow-Guard still trails always-on group-threshold on the strictest fairness metric.
- This should be acknowledged directly.

### 11. Conclusion

#### 11.1 Final Takeaway

- Deployment-time fairness mitigation should be selective and conservative.
- The current evidence supports FairFlow-Guard, not always-on RL, as the strongest contribution of the project.

#### 11.2 Future Work

- stronger activation calibration on hard datasets
- delayed-label deployment settings
- multiple protected attributes
- theoretical intervention-budget guarantees
- RL variants that optimize under a threshold-guard constraint rather than learning from scratch

## Planned Figures

1. Method diagram:
   - base model
   - rolling fairness monitor
   - activation gate
   - threshold-correction module
   - final decision

2. Accuracy vs DPR trade-off scatter:
   - show `base_model`, `group_threshold`, `guard_threshold`, `universal_rl`

3. Intervention vs fairness scatter:
   - this will clearly show the deployment trade-off

4. Order-stress robustness figure:
   - compare `guard_threshold` and `universal_rl` across protocols

5. Rolling trace plot:
   - `adult` and `compas` are the best candidates

## Planned Tables

1. Main benchmark table
2. Overall method summary table
3. Paired win-count table against `group_threshold`
4. Guard ablation table
5. Order-stress table
6. Guard diagnostics table

## Appendix Structure

### A. Dataset preprocessing details

### B. Hyperparameter grids

### C. Additional per-dataset and per-seed results

### D. Additional rolling-trace plots

### E. Reproducibility checklist

- hardware
- software versions
- seeds
- runtime notes
- dataset links
