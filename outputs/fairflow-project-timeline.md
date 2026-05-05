# FairFlow Project Timeline

## Purpose Of This Document

This document is a complete handoff of the FairFlow project evolution so far.
It is meant for:

- collaborators joining the project midstream,
- stronger research systems that need the full experiment history,
- future paper drafting and method redesign decisions.

It answers five questions:

1. What was the original project trying to do?
2. What code and methods were used in each phase?
3. What experiments were run and what did they show?
4. What changes were made after each disappointing result, and why?
5. Where exactly are we stuck now, and what are we trying to discover next?

---

## 1. Project Origin

### 1.1 Original Hackathon Vision

FairFlow started as a hackathon project built around a strong idea:

> Can we build a reinforcement-learning-based fairness layer that is both model-agnostic and dataset-agnostic?

The intended usage pattern was:

- take any deployed binary classifier,
- observe its prediction stream,
- monitor fairness online using rolling statistics,
- selectively override some predictions to maintain fairness,
- do this without retraining the base model.

This was not meant to be a new classifier. It was meant to be a fairness control layer around an existing classifier.

### 1.2 Core Original Technical Idea

The original innovation was the **fixed, dataset-agnostic state representation**.

Instead of feeding raw features to RL, FairFlow represents the current situation using a small vector of fairness and stream statistics:

- base prediction,
- base confidence,
- protected group,
- rolling demographic parity ratio,
- rolling TPR/FPR differences,
- group approval rates,
- intervention rate,
- group ratio,
- stream-order context,
- confidence gap.

This idea appears in the original hackathon code:

- [src/wrapper.py](/C:/Users/DELL/Desktop/hckton/ImpactThon/fairflow/src/wrapper.py)
- [src/environment/universal_fairness_env.py](/C:/Users/DELL/Desktop/hckton/ImpactThon/fairflow/src/environment/universal_fairness_env.py)
- [src/agents/train_universal.py](/C:/Users/DELL/Desktop/hckton/ImpactThon/fairflow/src/agents/train_universal.py)

### 1.3 What The Original Hackathon Code Already Had

- a universal wrapper around any base model,
- a 12-dimensional fixed RL state,
- a universal Gymnasium environment,
- PPO training on multiple synthetic fairness scenarios,
- a fallback threshold mode if RL could not be loaded.

### 1.4 What The Original Hackathon Code Did Not Yet Have

- a paper-grade benchmark,
- standardized multiple-dataset evaluation,
- strong baselines,
- multi-seed reporting,
- robust order-shift evaluation,
- clean separation between demo code and research code,
- a convincing novel method beyond the general RL-wrapper idea.

This is why the first major step after the grant was to build a serious benchmark stack around the hackathon code.

---

## 2. Research Framing Shift From Literature Review

Before writing experiments, the project was reframed using current research trends.

### 2.1 Initial Intended Research Story

The first paper idea was:

> A universal RL controller can improve fairness across datasets and models without retraining.

### 2.2 Why That Story Became Weak

The literature review showed that:

- post-processing fairness is now mainstream and quite strong,
- deployment-time fairness and runtime monitoring are increasingly important,
- fairness under shift is an active concern,
- RL fairness is promising but still comparatively brittle and hard to defend as a clean deployment solution.

That led to the first major scientific pivot:

> Do not center the paper on "RL solves fairness universally."
> Center it on "selective deployment-time fairness control for black-box classifiers."

This positioning is captured in:

- [outputs/elite-research-positioning.md](/C:/Users/DELL/Desktop/hckton/ImpactThon/fairflow/outputs/elite-research-positioning.md)

---

## 3. Phase 1: First Paper-Grade Benchmark Stack

### 3.1 Goal

Turn the hackathon idea into a reproducible research pipeline.

### 3.2 New Code Added

This phase created the first research pipeline:

- [research/runner.py](/C:/Users/DELL/Desktop/hckton/ImpactThon/fairflow/research/runner.py)
- [research/datasets.py](/C:/Users/DELL/Desktop/hckton/ImpactThon/fairflow/research/datasets.py)
- [research/dataset_catalog.py](/C:/Users/DELL/Desktop/hckton/ImpactThon/fairflow/research/dataset_catalog.py)
- [research/metrics.py](/C:/Users/DELL/Desktop/hckton/ImpactThon/fairflow/research/metrics.py)
- [research/baselines.py](/C:/Users/DELL/Desktop/hckton/ImpactThon/fairflow/research/baselines.py)
- [research/rl.py](/C:/Users/DELL/Desktop/hckton/ImpactThon/fairflow/research/rl.py)

Kaggle scripts:

- [kaggle/00_check_inputs.py](/C:/Users/DELL/Desktop/hckton/ImpactThon/fairflow/kaggle/00_check_inputs.py)
- [kaggle/01_main_benchmark.py](/C:/Users/DELL/Desktop/hckton/ImpactThon/fairflow/kaggle/01_main_benchmark.py)
- [kaggle/02_state_ablation.py](/C:/Users/DELL/Desktop/hckton/ImpactThon/fairflow/kaggle/02_state_ablation.py)
- [kaggle/03_reward_ablation.py](/C:/Users/DELL/Desktop/hckton/ImpactThon/fairflow/kaggle/03_reward_ablation.py)
- [kaggle/04_order_stress.py](/C:/Users/DELL/Desktop/hckton/ImpactThon/fairflow/kaggle/04_order_stress.py)
- [kaggle/05_make_paper_tables.py](/C:/Users/DELL/Desktop/hckton/ImpactThon/fairflow/kaggle/05_make_paper_tables.py)

### 3.3 Datasets Used

- Adult
- German Credit
- COMPAS
- Bank Marketing
- Recruitment

### 3.4 Models Used

- logistic regression
- random forest
- xgboost

### 3.5 Methods Evaluated

- `base_model`
- `group_threshold`
- `rule_based`
- `universal_rl`
- `dataset_specific_rl` for selected settings

### 3.6 Debugging Problems During This Phase

This phase included several important infrastructure fixes:

1. The first Kaggle run only processed `adult`, `german_credit`, and `recruitment`.
   - Cause: missing Kaggle input attachments for `compas` and `bank_marketing`.

2. Bank Marketing initially crashed.
   - Cause: Kaggle version used `deposit` instead of `y`.
   - Fix: dataset loader was updated to support both schemas.

3. COMPAS initially did not load.
   - Cause: Kaggle variant used a different CSV naming convention.
   - Fix: dataset loader was extended to support common Kaggle/ProPublica file variants.

4. PPO was trying to run on GPU in Kaggle for an MLP policy.
   - Fix: RL training/evaluation was forced to CPU where appropriate.

These were engineering fixes, not scientific method changes, but they were necessary before trusting the results.

### 3.7 Final Phase 1 Benchmark Output

Main result file:

- [research_outputs/main_benchmark/main_results.csv](/C:/Users/DELL/Desktop/hckton/ImpactThon/fairflow/research_outputs/main_benchmark/main_results.csv)

Final row count:

- `65` rows = `5 datasets x 3 models x 4 methods + 5 dataset_specific_rl rows`

### 3.8 Phase 1 Overall Results

Average performance across the main benchmark:

| Method | Accuracy | DPR | EO Gap | Intervention | Fairness Pass Rate |
|---|---:|---:|---:|---:|---:|
| `base_model` | 0.8256 | 0.7813 | 0.0806 | 0.0000 | 0.60 |
| `group_threshold` | 0.8127 | 0.9361 | 0.0923 | 0.0634 | 0.80 |
| `rule_based` | 0.8208 | 0.8526 | 0.0569 | 0.0180 | 0.80 |
| `universal_rl` | 0.7895 | 1.1495 | 0.1839 | 0.0727 | 1.00 |
| `dataset_specific_rl` | 0.8120 | 1.0320 | 0.0618 | 0.0450 | 1.00 |

On the hardest fairness datasets (`adult` + `compas`):

| Method | Accuracy | DPR | EO Gap | Intervention |
|---|---:|---:|---:|---:|
| `base_model` | 0.7913 | 0.4707 | 0.1501 | 0.0000 |
| `group_threshold` | 0.7810 | 0.7749 | 0.1342 | 0.0682 |
| `rule_based` | 0.7792 | 0.6490 | 0.0911 | 0.0450 |
| `universal_rl` | 0.7324 | 1.0447 | 0.1469 | 0.1251 |
| `dataset_specific_rl` | 0.7492 | 1.0704 | 0.0990 | 0.1077 |

### 3.9 What Phase 1 Taught Us

Good news:

- the project was real,
- fairness could be improved,
- the universal RL idea was not nonsense,
- the fixed-state representation was usable.

Bad news:

- `universal_rl` improved fairness, but at high intervention and with large accuracy cost,
- `dataset_specific_rl` often had a better trade-off than `universal_rl`,
- the universal claim was too ambitious,
- the method was not yet a clean paper story.

### 3.10 Phase 1 Ablation Findings

State ablation table:

- [research_outputs/paper_tables/table_state_ablation.csv](/C:/Users/DELL/Desktop/hckton/ImpactThon/fairflow/research_outputs/paper_tables/table_state_ablation.csv)

Reward ablation table:

- [research_outputs/paper_tables/table_reward_ablation.csv](/C:/Users/DELL/Desktop/hckton/ImpactThon/fairflow/research_outputs/paper_tables/table_reward_ablation.csv)

Key findings:

- Removing fairness-rate features collapsed fairness behavior.
  - `no_fairness_rates` on Adult: DPR `0.3243`
  - `no_fairness_rates` on COMPAS: DPR `0.6232`
- Fairness-heavy rewards improved fairness but at larger intervention and larger EO distortion.
  - `fairness_heavy` on COMPAS: DPR `1.0708`, intervention `0.1858`
- Accuracy-heavy rewards preserved utility but under-corrected fairness.

### 3.11 Phase 1 Order-Stress Findings

Order-stress table:

- [research_outputs/paper_tables/table_order_stress.csv](/C:/Users/DELL/Desktop/hckton/ImpactThon/fairflow/research_outputs/paper_tables/table_order_stress.csv)

The universal RL controller was very sensitive to stream ordering:

- Adult `privileged_first`: DPR `2.5962`
- Adult `unprivileged_first`: DPR `3.2556`
- Recruitment `privileged_first`: DPR `2.6347`
- Recruitment `unprivileged_first`: DPR `3.0386`

### 3.12 Why We Pivoted After Phase 1

At this point the right scientific conclusion was:

> "Universal RL can repair unfairness, but it over-corrects and is brittle under stream shift."

That is interesting, but not yet a best-paper-level result.

So we changed the problem statement.

---

## 4. Phase 2: Selective Fairness Guard Reframing

### 4.1 New Problem Statement

Instead of asking:

> Can a universal RL agent fix fairness for everything?

we asked:

> Can a selective deployment-time controller activate fairness correction only when needed, reduce unnecessary interventions, and remain stable under stream order shift?

This was a much better fit to both the literature and the observed weaknesses of always-on RL.

### 4.2 New Code Added

This was the first "elite" redesign:

- [research/elite_methods.py](/C:/Users/DELL/Desktop/hckton/ImpactThon/fairflow/research/elite_methods.py)
- [research/elite_runner.py](/C:/Users/DELL/Desktop/hckton/ImpactThon/fairflow/research/elite_runner.py)
- [research/statistics.py](/C:/Users/DELL/Desktop/hckton/ImpactThon/fairflow/research/statistics.py)

New Kaggle scripts:

- [kaggle/06_elite_benchmark.py](/C:/Users/DELL/Desktop/hckton/ImpactThon/fairflow/kaggle/06_elite_benchmark.py)
- [kaggle/07_guard_ablation.py](/C:/Users/DELL/Desktop/hckton/ImpactThon/fairflow/kaggle/07_guard_ablation.py)
- [kaggle/08_elite_order_stress.py](/C:/Users/DELL/Desktop/hckton/ImpactThon/fairflow/kaggle/08_elite_order_stress.py)
- [kaggle/09_make_elite_tables.py](/C:/Users/DELL/Desktop/hckton/ImpactThon/fairflow/kaggle/09_make_elite_tables.py)

### 4.3 Methods Introduced

- `guard_threshold`
  - a hysteretic selective controller that activates threshold-based correction only when rolling fairness drops
- `fairflow_guard_rl`
  - a conservative RL gate over threshold-based candidate corrections

Existing baselines retained:

- `base_model`
- `group_threshold`
- `rule_based`
- `universal_rl`

### 4.4 Why These Changes Were Made

They directly addressed the Phase 1 failure modes:

- If always-on RL was too aggressive, try a selective controller.
- If static threshold correction was strong but heavy-handed, try to activate it only when needed.
- If order stress broke RL, add hysteresis and bounded activation windows.

### 4.5 Phase 2 Benchmark Output

Main result folder:

- [research_outputs/elite_benchmark](/C:/Users/DELL/Desktop/hckton/ImpactThon/fairflow/research_outputs/elite_benchmark)

Row counts:

- benchmark: `180` rows = `5 datasets x 2 models x 6 methods x 3 seeds`
- guard ablation: `54` rows
- order stress: `144` rows

### 4.6 Phase 2 Overall Results

From [research_outputs/elite_benchmark/method_summary.csv](/C:/Users/DELL/Desktop/hckton/ImpactThon/fairflow/research_outputs/elite_benchmark/method_summary.csv):

| Method | Accuracy | DPR | EO Gap | Intervention | Tail Fairness | Fairness Pass |
|---|---:|---:|---:|---:|---:|---:|
| `base_model` | 0.8140 | 0.7707 | 0.1060 | 0.0000 | 0.4751 | 0.60 |
| `group_threshold` | 0.8048 | 0.9576 | 0.1037 | 0.0688 | 0.6905 | 0.933 |
| `guard_threshold` | 0.8077 | 0.8920 | 0.0875 | 0.0422 | 0.6129 | 0.767 |
| `rule_based` | 0.8088 | 0.8479 | 0.0727 | 0.0205 | 0.5810 | 0.80 |
| `fairflow_guard_rl` | 0.8113 | 0.8288 | 0.0792 | 0.0136 | 0.5442 | 0.60 |
| `universal_rl` | 0.7954 | 0.9930 | 0.0818 | 0.0461 | 0.7689 | 0.867 |

### 4.7 What Phase 2 Showed

This phase produced an important scientific refinement:

- `group_threshold` was the strongest fairness baseline.
- `guard_threshold` was the strongest **practical selective controller**.
- `fairflow_guard_rl` was too conservative.
- `universal_rl` still had strong fairness numbers, but remained a deployment-risk method because of shift brittleness.

### 4.8 Phase 2 Guard Ablation Findings

From [research_outputs/elite_tables/table_guard_ablation.csv](/C:/Users/DELL/Desktop/hckton/ImpactThon/fairflow/research_outputs/elite_tables/table_guard_ablation.csv):

Average across ablation datasets:

| Method | Accuracy | DPR | EO Gap | Intervention | Tail Fairness |
|---|---:|---:|---:|---:|---:|
| `fairflow_guard_rl` | 0.8519 | 0.7017 | 0.0829 | 0.0157 | 0.3708 |
| `group_threshold` | 0.8469 | 0.8754 | 0.1073 | 0.0567 | 0.5840 |
| `guard_rl_no_anchor` | 0.8311 | 0.8733 | 0.0628 | 0.0491 | 0.6023 |
| `guard_threshold` | 0.8498 | 0.7979 | 0.0936 | 0.0402 | 0.4900 |
| `guard_threshold_no_hysteresis` | 0.8487 | 0.7899 | 0.0953 | 0.0390 | 0.4719 |
| `universal_rl` | 0.8274 | 0.9163 | 0.0737 | 0.0611 | 0.6662 |

Interpretation:

- hysteresis helped somewhat,
- threshold anchoring mattered,
- RL gating without a strong anchor was not enough,
- simple guard logic remained stronger than RL-based selectivity.

### 4.9 Phase 2 Order-Stress Findings

From [research_outputs/elite_order_stress/aggregated_results.csv](/C:/Users/DELL/Desktop/hckton/ImpactThon/fairflow/research_outputs/elite_order_stress/aggregated_results.csv):

Average across the order-stress benchmark:

| Method | Protocol | Accuracy | DPR | Tail Fairness |
|---|---|---:|---:|---:|
| `universal_rl` | natural | 0.8274 | 0.9163 | 0.6662 |
| `universal_rl` | privileged_first | 0.7242 | 1.8955 | 0.0087 |
| `universal_rl` | unprivileged_first | 0.7163 | 1.9424 | 0.0086 |
| `guard_threshold` | natural | 0.8498 | 0.7979 | 0.4900 |
| `guard_threshold` | privileged_first | 0.8498 | 0.7846 | 0.0028 |
| `guard_threshold` | unprivileged_first | 0.8502 | 0.7852 | 0.0034 |
| `fairflow_guard_rl` | natural | 0.8519 | 0.7017 | 0.3708 |
| `fairflow_guard_rl` | privileged_first | 0.8514 | 0.7219 | 0.0009 |
| `fairflow_guard_rl` | unprivileged_first | 0.8517 | 0.7200 | 0.0025 |

Interpretation:

- `universal_rl` remained highly brittle under adversarial ordering.
- `guard_threshold` was much more stable.
- This is why the paper framing shifted away from always-on RL.

### 4.10 Why We Did Not Stop Here

This phase still had a major scientific weakness:

- `group_threshold` was only a baseline, not a novel contribution,
- `guard_threshold` was better framed as a controller around a baseline,
- `fairflow_guard_rl` was not strong enough to headline.

So the next step was to try a genuinely new adaptive method.

---

## 5. Phase 3: Adaptive Guard Introduction

### 5.1 Motivation

We wanted a method that was more novel than `guard_threshold`.

Instead of a simple trigger-and-hysteresis controller, we introduced a per-step projected decision rule:

> Compare the base decision and a threshold-corrected candidate online. Intervene only if the projected fairness benefit is worth the expected utility cost.

### 5.2 New Method

- `adaptive_guard`

Implemented in:

- [research/elite_methods.py](/C:/Users/DELL/Desktop/hckton/ImpactThon/fairflow/research/elite_methods.py)

Rerun folder:

- [elite results adaptive](/C:/Users/DELL/Desktop/hckton/ImpactThon/fairflow/elite results adaptive)

### 5.3 Why This Seemed Promising

The method had a plausible novelty story:

- online,
- projected fairness-utility trade-off,
- lower-touch than always-on correction,
- more principled than simple if-threshold-then-activate logic.

### 5.4 Phase 3 Overall Results

From [elite results adaptive/elite_benchmark/method_summary.csv](/C:/Users/DELL/Desktop/hckton/ImpactThon/fairflow/elite%20results%20adaptive/elite_benchmark/method_summary.csv):

| Method | Accuracy | DPR | EO Gap | Intervention |
|---|---:|---:|---:|---:|
| `adaptive_guard` | 0.8127 | 0.7981 | 0.0865 | 0.0130 |
| `guard_threshold` | 0.8077 | 0.8920 | 0.0875 | 0.0422 |
| `group_threshold` | 0.8048 | 0.9576 | 0.1037 | 0.0688 |
| `universal_rl` | 0.7954 | 0.9930 | 0.0818 | 0.0461 |

### 5.5 What Went Wrong

`adaptive_guard` was **too conservative**.

It protected utility, but it did not fix fairness enough.

Its diagnostics from [elite results adaptive/elite_benchmark/guard_diagnostics.csv](/C:/Users/DELL/Desktop/hckton/ImpactThon/fairflow/elite%20results%20adaptive/elite_benchmark/guard_diagnostics.csv) show why:

- activation rate mean: `0.0419`
- candidate rate mean: `0.0688`
- accept rate mean: `0.0130`
- accept-given-candidate mean: `0.1851`
- average projected DPR gain: `-0.0333`

Interpretation:

- the controller saw candidate corrections,
- but accepted too few of them,
- and the scoring rule was too utility-dominant or misaligned.

### 5.6 Scientific Takeaway From Phase 3

The method idea was interesting, but the current objective was wrong.

It behaved like:

- "avoid cost,"
- "avoid intervention,"
- "repair unfairness only occasionally,"

instead of:

- "repair fairness aggressively when the system is clearly unfair, then become conservative again."

That led to the next redesign.

---

## 6. Phase 4: Deficit-Focused Adaptive Guard

### 6.1 Motivation

The next hypothesis was:

> The adaptive controller fails because it treats fairness repair and overshoot control too symmetrically.

So the next redesign made the controller **asymmetric**:

- aggressive when DPR is below threshold,
- conservative once the stream is in a safe band.

### 6.2 What Was Changed In Code

Still in [research/elite_methods.py](/C:/Users/DELL/Desktop/hckton/ImpactThon/fairflow/research/elite_methods.py), the adaptive controller was changed to:

- separate **deficit mode** from **safe mode**,
- require candidate interventions to improve projected DPR in deficit mode,
- allow bounded utility sacrifice while fairness is below threshold,
- use stronger fairness weighting,
- use conservative release behavior in the safe region.

The current parameter grids reflect that redesign:

- `deficit_weight_grid = (4.0, 6.0, 8.0)`
- `overshoot_weight_grid = (0.75, 1.25)`
- `deficit_utility_slack_grid = (0.05, 0.10, 0.15)`
- `safe_utility_slack_grid = (0.0, 0.02)`
- `safe_min_gain_grid = (0.02, 0.05)`
- `deficit_min_improvement_grid = (0.0, 0.01)`

These were intended to make the controller intervene more on genuinely unfair streams while staying low-touch elsewhere.

### 6.3 What We Expected

We expected:

- higher DPR on `adult` and `compas`,
- more accepted corrections on unfair datasets,
- still-low intervention on already-fair datasets,
- better novelty story than `guard_threshold`.

### 6.4 What Actually Happened

After rerunning the adaptive bundle, the current results still look materially weak.

The latest adaptive benchmark in [elite results adaptive](/C:/Users/DELL/Desktop/hckton/ImpactThon/fairflow/elite%20results%20adaptive) still shows:

- `adaptive_guard` accuracy `0.8127`
- `adaptive_guard` DPR `0.7981`
- `adaptive_guard` EO gap `0.0865`
- `adaptive_guard` intervention `0.0130`

And under order stress:

| Method | Protocol | Accuracy | DPR | Tail Fairness |
|---|---|---:|---:|---:|
| `adaptive_guard` | natural | 0.8534 | 0.6864 | 0.3455 |
| `adaptive_guard` | privileged_first | 0.8552 | 0.6311 | 0.0012 |
| `adaptive_guard` | unprivileged_first | 0.8553 | 0.6308 | 0.0018 |

### 6.5 Did It Work?

Not enough.

The controller still behaves like a low-touch utility-preserving policy, not like a fairness-repair policy.

It did **not** become the clear proposed method we wanted.

### 6.6 Important Ambiguity

Because the adaptive folder was reused and the post-patch results remained very similar to the pre-patch adaptive analysis, there is a mild execution ambiguity:

- either the newest logic still converged to nearly the same behavior,
- or the rerun did not meaningfully alter the selected parameter regime.

Regardless, the **scientific conclusion does not change**:

> the adaptive controller family is still too conservative and still not strong enough to carry the paper.

---

## 7. Code And Experiment Timeline Summary

### Stage A: Original hackathon implementation

Core code:

- `src/wrapper.py`
- `src/environment/universal_fairness_env.py`
- `src/agents/train_universal.py`

Main idea:

- universal RL wrapper with fixed fairness state

Status:

- compelling concept, not paper-ready

### Stage B: First research benchmark

Scripts:

- `00_check_inputs.py`
- `01_main_benchmark.py`
- `02_state_ablation.py`
- `03_reward_ablation.py`
- `04_order_stress.py`
- `05_make_paper_tables.py`

Methods:

- base model
- group threshold
- rule based
- universal RL
- dataset-specific RL

Outcome:

- strong evidence that fairness mitigation works
- evidence that universal RL is over-aggressive and brittle

### Stage C: Selective guard redesign

Scripts:

- `06_elite_benchmark.py`
- `07_guard_ablation.py`
- `08_elite_order_stress.py`
- `09_make_elite_tables.py`

Methods added:

- `guard_threshold`
- `fairflow_guard_rl`

Outcome:

- `group_threshold` strongest fairness baseline
- `guard_threshold` strongest practical selective method
- `fairflow_guard_rl` too conservative

### Stage D: Adaptive method attempt

Methods added:

- `adaptive_guard`

Outcome:

- more novel
- still too conservative
- not enough fairness repair on the unfair datasets

### Stage E: Deficit-focused adaptive redesign

Change:

- asymmetric deficit-vs-safe controller

Outcome:

- still not strong enough
- project remains scientifically interesting but method is still not "the one"

---

## 8. Where We Are Currently Stuck

We are stuck on the core scientific problem:

> We still do not have a clearly novel method that is both strong enough and stable enough to beat the framing problem created by the strength of the `group_threshold` baseline.

### 8.1 The Main Technical Bottlenecks

1. **The fairness baseline is very strong.**
   - `group_threshold` is hard to beat on fairness repair.
   - Any new method must justify itself by either improving robustness, reducing intervention, or offering a deeper theoretical contribution.

2. **Always-on RL is too brittle.**
   - It can improve fairness, but it becomes unstable under order shift.
   - That makes it hard to defend as a deployment-safe main method.

3. **Selective RL has been too conservative.**
   - `fairflow_guard_rl` does not accept enough fairness corrections.
   - `adaptive_guard` also accepts too few interventions.

4. **The novelty-efficiency trade-off is unresolved.**
   - Simple guards are operationally good, but risk being seen as engineering wrappers.
   - More novel adaptive methods have not yet delivered stronger empirical results.

5. **The hard datasets remain the decisive test.**
   - `adult` and `compas` are where the method must prove itself.
   - A method that looks elegant overall but does not repair fairness there will not carry the paper.

### 8.2 The Scientific Reason We Do Not Yet Have The Result We Wanted

So far, every method family has failed in one of two ways:

- **always-on methods** repair fairness but over-intervene or become unstable,
- **selective methods** are stable and efficient but under-correct fairness.

What we want is a controller that does **both**:

- repair fairness aggressively when needed,
- stay quiet when the stream is already acceptable.

We do not yet have that.

---

## 9. What We Are Actually Trying To Find Next

The next successful method must satisfy five properties at once.

### 9.1 Method Requirements

1. **Model agnostic**
   - must wrap a black-box classifier

2. **Strong on unfair datasets**
   - must substantially improve fairness on `adult` and `compas`

3. **Low unnecessary intervention**
   - must stay quiet on already-fair datasets like `german_credit`, `bank_marketing`, and many recruitment settings

4. **Robust to stream order**
   - must not collapse under `privileged_first` or `unprivileged_first`

5. **Scientifically novel**
   - must be more than "use a standard baseline only when metric X is low"

### 9.2 The Specific Research Object We Are Looking For

We are trying to discover a method that can be described as:

> an online, model-agnostic, deployment-time fairness controller with principled selective intervention and stable behavior under shift.

### 9.3 The Most Plausible Next Direction

The most plausible next family is a more principled controller, for example:

- an online primal-dual fairness controller,
- a constrained optimization controller with rolling fairness budget,
- a threshold-offset controller with fairness deficit multipliers,
- a selective control-law formulation rather than a heuristic accept/reject rule.

Why this direction matters:

- it is more novel than static thresholding,
- more controllable than unrestricted RL,
- more principled than the current adaptive heuristic,
- easier to justify scientifically if it comes with a clear objective or constraint interpretation.

---

## 10. Recommended Questions For Colleagues Or Stronger Research Systems

These are the best questions to ask next:

1. What online control formulation can enforce or approximately enforce rolling fairness constraints with minimal intervention?
2. Is there a principled primal-dual or constrained bandit/controller formulation better suited than PPO for this deployment problem?
3. How should fairness deficit be defined online so that the controller is aggressive when unfair and conservative when safe?
4. Can the threshold-correction baseline be converted into a learnable threshold-offset control law with guarantees or monotonicity?
5. What novelty claim is strongest here:
   - universal RL fairness,
   - selective fairness guardrails,
   - online constrained fairness control,
   - runtime fairness stabilization under shift?

---

## 11. Key Artifact Map

Original benchmark outputs:

- [research_outputs/main_benchmark](/C:/Users/DELL/Desktop/hckton/ImpactThon/fairflow/research_outputs/main_benchmark)
- [research_outputs/paper_tables](/C:/Users/DELL/Desktop/hckton/ImpactThon/fairflow/research_outputs/paper_tables)

Selective-guard benchmark outputs:

- [research_outputs/elite_benchmark](/C:/Users/DELL/Desktop/hckton/ImpactThon/fairflow/research_outputs/elite_benchmark)
- [research_outputs/elite_guard_ablation](/C:/Users/DELL/Desktop/hckton/ImpactThon/fairflow/research_outputs/elite_guard_ablation)
- [research_outputs/elite_order_stress](/C:/Users/DELL/Desktop/hckton/ImpactThon/fairflow/research_outputs/elite_order_stress)
- [research_outputs/elite_tables](/C:/Users/DELL/Desktop/hckton/ImpactThon/fairflow/research_outputs/elite_tables)

Adaptive benchmark outputs:

- [elite results adaptive](/C:/Users/DELL/Desktop/hckton/ImpactThon/fairflow/elite%20results%20adaptive)

Current paper structure draft:

- [paper-outline.md](/C:/Users/DELL/Desktop/hckton/ImpactThon/fairflow/paper-outline.md)

Research positioning note:

- [outputs/elite-research-positioning.md](/C:/Users/DELL/Desktop/hckton/ImpactThon/fairflow/outputs/elite-research-positioning.md)

---

## 12. Bottom Line

The project is not failing because the idea is bad.

The project is stuck because the original big idea:

> "universal RL fairness control"

was interesting but too weak as a finished scientific method, and every redesign so far has exposed a hard trade-off:

- fairness repair,
- intervention cost,
- stability under shift,
- and novelty.

The next step is not to run more of the same experiments.

The next step is to identify a new **control formulation** that can:

- match enough of the fairness gains of `group_threshold`,
- keep interventions materially lower,
- avoid the instability of always-on RL,
- and be defensible as a new research contribution.

