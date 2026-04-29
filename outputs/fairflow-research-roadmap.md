# FairFlow Research Roadmap

Date: 2026-04-28

## 1. What this project is trying to be

After reading the repository, the clearest interpretation is:

**FairFlow is a post-deployment fairness controller for black-box classifiers.**

It is not just a fair classifier and not just a dashboard. The core idea is:

- keep an already-trained model unchanged,
- observe its prediction stream,
- maintain rolling fairness statistics,
- let a reinforcement-learning policy decide when to keep, flip, or soften the model's decision,
- improve fairness without retraining the underlying model.

The technically interesting part is the claim that this controller is **dataset-agnostic and model-agnostic** because it operates on a fixed-dimensional state of streaming statistics rather than raw task-specific features.

Repository evidence:

- [src/wrapper.py](C:\Users\DELL\Desktop\hckton\ImpactThon\fairflow\src\wrapper.py) describes a wrapper for "ANY machine learning model" and says it should work on "any dataset without retraining".
- [src/environment/universal_fairness_env.py](C:\Users\DELL\Desktop\hckton\ImpactThon\fairflow\src\environment\universal_fairness_env.py) explicitly calls the dataset-agnostic state representation the "key innovation".
- [src/agents/train_universal.py](C:\Users\DELL\Desktop\hckton\ImpactThon\fairflow\src\agents\train_universal.py) trains a universal PPO agent across multiple synthetic scenarios.
- [backend/main.py](C:\Users\DELL\Desktop\hckton\ImpactThon\fairflow\backend\main.py) wraps this into a demo system with live metrics, switching across datasets/models, and compliance-style monitoring.

## 2. What is already promising

The project has real research potential, but only under a narrower framing than the hackathon pitch.

The strongest publishable angle is:

**A universal post-deployment fairness controller for streaming binary classification under distribution shift.**

Why this angle is credible:

- It is black-box: the base model can remain untouched.
- It is intervention-based: the policy acts only when needed.
- It is transfer-oriented: the same controller can be applied across datasets and model families.
- It is sequential: fairness is monitored and acted upon over time, not only once at training time.
- It naturally connects three active research threads:
  - fair sequential decision making,
  - fairness under drift/distribution shift,
  - post-processing fairness for deployed models.

## 3. What the current code already demonstrates

### 3.1 Strong signal

The current implementation already contains a real, non-trivial research prototype:

- fixed universal state with `STATE_DIM = 12` in both the wrapper and universal environment,
- PPO policy trained across varying scenarios,
- multiple datasets and base models,
- online fairness statistics and intervention tracking,
- a clean black-box wrapper interface.

### 3.2 Empirical signal from the current repo

I evaluated the saved universal PPO agent across the included datasets and models.

High-level outcome:

- On `adult`, the controller meaningfully improves demographic parity ratio with modest accuracy loss.
- On `german`, the controller mostly stays inactive because the base models are already relatively fair.
- On `recruitment`, it makes small, low-intervention fairness improvements while preserving high accuracy.

This is exactly the kind of behavior you want for a deployment controller:

- intervene when bias is present,
- stay low-touch when the base model is already acceptable.

That is a much stronger story than "always changes predictions a lot".

### 3.3 The current strongest claim

The most defensible claim today is:

**A fairness-aware RL wrapper trained on domain-invariant statistical states can transfer across multiple datasets and base models, improving group fairness with relatively low intervention rates.**

## 4. What is not yet research-ready

This part matters a lot. Right now, several pieces are still hackathon-grade rather than paper-grade.

### 4.1 Placeholder online state variables

In [src/wrapper.py](C:\Users\DELL\Desktop\hckton\ImpactThon\fairflow\src\wrapper.py), `_calculate_rate_diffs()` currently returns zeros because true labels are unavailable online. That means TPR/FPR-difference features are placeholders in actual deployment use.

In [backend/main.py](C:\Users\DELL\Desktop\hckton\ImpactThon\fairflow\backend\main.py), parts of `get_universal_state()` also use placeholder values for TPR difference, FPR difference, and consecutive same-group.

This is acceptable for a demo, but a reviewer will immediately ask:

- what information is truly available at decision time?
- are you using delayed labels, proxy estimates, or unavailable oracle statistics?

### 4.2 Heavy reliance on synthetic training

The universal agent is trained mainly on synthetic scenarios in [src/agents/train_universal.py](C:\Users\DELL\Desktop\hckton\ImpactThon\fairflow\src\agents\train_universal.py).

That is not a flaw by itself, but it changes the paper claim. The claim cannot be:

**"We learned universality from real-world domains."**

The claim becomes:

**"We use synthetic curriculum scenarios to learn a transferable fairness controller, then evaluate transfer to real datasets."**

That is still publishable, but it must be stated honestly.

### 4.3 Demo hardcoding

[backend/main.py](C:\Users\DELL\Desktop\hckton\ImpactThon\fairflow\backend\main.py) contains hardcoded demo values and precomputed cases. These are fine for product storytelling, but they cannot be mixed with research evidence.

### 4.4 Reproducibility issues

The saved PPO artifact needed compatibility workarounds to load due to dependency serialization mismatches. Some sklearn models also warn about version mismatches.

For a paper, you need a clean reproduction path.

## 5. Literature landscape, 2021-2025

The literature is active, but it is split across adjacent subproblems. FairFlow sits at the intersection of all of them.

### 5.1 Theme A: Fair reinforcement learning and long-term fairness

Recent surveys show this area is still early-stage and fragmented.

- Reuel and Ma, *Fairness in Reinforcement Learning: A Survey* (2024): highlights that fair RL remains nascent and explicitly calls for cross-domain fair RL and fairness throughout runtime rather than only at the terminal outcome.
  - https://arxiv.org/abs/2405.06909
- Gohar et al., *Long-Term Fairness Inquiries and Pursuits in Machine Learning* (2024, accepted in TMLR): argues that static fairness methods often fail in dynamic settings with feedback loops and identifies dynamic modeling, distribution shift, and performative effects as core settings.
  - https://arxiv.org/abs/2406.06736
- Hu and Zhang, *Achieving Long-Term Fairness in Sequential Decision Making* (AAAI 2022): formulates long-term fairness via causal interventions and constrained optimization.
  - https://arxiv.org/abs/2204.01819
- Yin et al., *Long-Term Fairness with Unknown Dynamics* (2023): studies online RL with unknown dynamics and fairness-violation guarantees.
  - https://arxiv.org/abs/2304.09362
- Hu et al., *Long-Term Fair Decision Making through Deep Generative Models* (AAAI 2024): uses temporal causal models and deep generative modeling for long-term fairness.
  - https://arxiv.org/abs/2401.11288
- Lear and Zhang, *A Causal Lens for Learning Long-term Fair Policies* (ICLR 2025): balances short-term and long-term fairness through causal decomposition.
  - https://openreview.net/forum?id=rPkCVSsoM4
- Hu, Lear, and Zhang, *Striking a Balance in Fairness for Dynamic Systems Through Reinforcement Learning* (2024): explicitly studies the tension between traditional fairness notions, long-term fairness, and utility.
  - https://arxiv.org/abs/2401.06318

What these papers usually emphasize:

- fairness in dynamic systems is not equivalent to static fairness,
- there are unavoidable trade-offs between utility and fairness,
- causal modeling often provides the cleanest formalism,
- general cross-domain fair RL remains underdeveloped.

### 5.2 Theme B: Fairness under distribution shift and drift

This is directly relevant to your deployment-controller story.

- Shao et al., *Supervised Algorithmic Fairness in Distribution Shifts: A Survey* (2024): reviews fairness under covariate, label, concept, and conditional shifts, and explicitly notes the lack of datasets tailored to fairness-under-shift evaluation.
  - https://arxiv.org/abs/2402.01327
- Deho et al., *Is it Still Fair? A Comparative Evaluation of Fairness Algorithms through the Lens of Covariate Drift* (2024): shows that drift can seriously deteriorate fairness and that the size/direction of drift is not a reliable proxy for fairness degradation.
  - https://arxiv.org/abs/2409.12428
- Davis et al., *Emerging algorithmic bias: fairness drift as the next dimension of model maintenance and sustainability* (JAMIA 2025): frames fairness drift as an ongoing post-deployment maintenance problem rather than a one-time model-development problem.
  - https://academic.oup.com/jamia/article/32/5/845/8074959

This part of the literature is a very good fit for FairFlow.

### 5.3 Theme C: Post-processing and black-box fairness

This is the closest static-literature neighbor to what you built.

- Padh et al., *Addressing fairness in classification with a model-agnostic multi-objective algorithm* (UAI 2021): model-agnostic, multi-objective fairness optimization for statistical-parity-style notions.
  - https://proceedings.mlr.press/v161/padh21a.html
- Xian et al., *Fair and Optimal Classification via Post-Processing* (ICML 2023): gives a principled post-processing view of demographic-parity tradeoffs and optimal fair post-processing from scores.
  - https://proceedings.mlr.press/v202/xian23b.html
- Xian and Zhao, *A Unified Post-Processing Framework for Group Fairness in Classification* (2024): unifies several group fairness criteria within one post-processing framework.
  - https://arxiv.org/abs/2405.04025
- Tifrea et al., *FRAPPÉ: A Group Fairness Framework for Post-Processing Everything* (ICML 2024): turns regularized in-processing methods into post-processing methods and argues for modular mitigation in black-box settings.
  - https://proceedings.mlr.press/v235/tifrea24a.html
- Zhang, Roth, and Zhang, *Fair Risk Control* (ICML 2024): post-processing fairness guarantees through generalized multicalibration.
  - https://proceedings.mlr.press/v235/zhang24be.html
- Cohen-Inger et al., *BiasGuard: Guardrailing Fairness in Machine Learning Production Systems* (2025): very close in spirit to your deployment story, but uses test-time augmentation and CTGAN rather than sequential RL control.
  - https://arxiv.org/abs/2501.04142

This cluster of papers is important because it means your paper should not oversell "model-agnostic fairness" as novel by itself. That part is already crowded.

### 5.4 Theme D: Runtime monitoring and fairness in the wild

- Henzinger et al., *Runtime Monitoring of Dynamic Fairness Properties* (FAccT 2023): argues that fairness should be evaluated at runtime, especially when the system and population evolve.
  - https://arxiv.org/abs/2305.04699
- Feng et al., *Designing monitoring strategies for deployed machine learning algorithms* (CLeaR 2024): focuses on post-deployment monitoring under performative effects and dynamic environments.
  - https://proceedings.mlr.press/v236/feng24a.html

This literature supports your operational framing:

- monitor fairness online,
- react online,
- do not assume training-time guarantees survive deployment.

## 6. What gap FairFlow can fill

The paper gap is not:

- "fairness-aware ML exists, and we also improve fairness";
- "RL can optimize fairness";
- "post-processing can be model-agnostic".

Those are already established.

The more interesting and still underexplored gap is:

**Can we build a universal, low-touch, post-deployment fairness controller that uses only deployment-time summary statistics, transfers across datasets and model families, and remains effective under streaming distribution shifts?**

That is where your implementation is distinctive.

## 7. Recommended paper statement

### Main problem statement

**We study post-deployment fairness control for streaming binary classification systems when the base model cannot be retrained or modified. We ask whether a reinforcement-learning controller operating on a dataset-agnostic state of online fairness statistics can generalize across datasets and classifier families, improving group fairness with minimal utility loss and limited intervention.**

### Cleaner title direction

Possible titles:

- **FairFlow: A Universal RL Controller for Post-Deployment Fairness in Streaming Classification**
- **Model-Agnostic Fairness Control Under Drift: A Reinforcement Learning Wrapper for Deployed Classifiers**
- **Learning When to Intervene: A Transferable RL Fairness Guardrail for Black-Box Classifiers**

## 8. Recommended methodology

### 8.1 Core paper formulation

Formulate the system as a sequential decision problem:

- base classifier emits score and label,
- environment exposes protected-group membership and rolling fairness statistics,
- controller action is one of:
  - keep prediction,
  - flip to positive,
  - flip to negative,
  - or a smaller action set if you simplify,
- reward balances:
  - predictive correctness,
  - fairness target attainment,
  - intervention penalty.

### 8.2 What to emphasize as the main contribution

Contribution 1:
**A universal state representation for post-deployment fairness control.**

Contribution 2:
**A transferable RL controller trained on synthetic fairness scenarios and deployed on real datasets/models without retraining the base classifier.**

Contribution 3:
**A benchmark protocol for fairness control under stream-order and drift stress tests.**

### 8.3 What to de-emphasize

Do not make the primary novelty claim:

- dashboard,
- SHAP explanations,
- compliance wording,
- generic fairness mitigation,
- "works for any dataset" without qualification.

Those can be supporting pieces, not the core research contribution.

## 9. Experiments you need

### 9.1 Main comparisons

At minimum compare:

1. Base model with no mitigation
2. Static threshold or post-processing baseline
3. Rule-based FairFlow fallback
4. Dataset-specific RL controller
5. Universal RL controller

If possible add:

6. FRAPPÉ-style or LinearPost-style post-processing baseline
7. BiasGuard-style deployment baseline if code is available or easily reproducible

### 9.2 Datasets

Keep the current three, but add at least one or two more public tabular fairness benchmarks if possible.

Recommended minimum:

- Adult
- German Credit
- Recruitment dataset from the repo
- COMPAS or ACS Income
- Bank Marketing or Law School if compatible

### 9.3 Base models

Use at least:

- Logistic Regression
- Random Forest
- XGBoost or LightGBM

This is important because the black-box transfer claim is one of your strongest assets.

### 9.4 Metrics

Report:

- accuracy
- balanced accuracy
- F1 and AUC if meaningful
- demographic parity ratio or difference
- equal opportunity difference
- equalized odds components: TPR gap and FPR gap
- intervention rate
- fairness-over-time metrics on rolling windows
- optionally calibration if scores matter

### 9.5 Stress tests

This is where your paper can become much stronger than a standard fairness paper.

Run:

- random stream shuffles
- privileged-first ordering
- unprivileged-first ordering
- bursty group-order drift
- class-prior drift
- changing minority-group ratio
- drifted score calibration or threshold shifts

Your preliminary results already suggest:

- random shuffles are fairly stable,
- extreme stream order can break the fairness target.

That is useful and publishable because it identifies a real limitation instead of hiding it.

## 10. Ablation studies you need

These are essential.

1. **State ablation**
   - remove confidence features
   - remove group-ratio features
   - remove intervention history
   - remove delayed fairness estimates

2. **Reward ablation**
   - vary fairness weight
   - vary intervention penalty
   - compare sparse vs dense fairness reward

3. **Training distribution ablation**
   - synthetic-only
   - synthetic + real fine-tuning
   - curriculum on vs off

4. **Transfer ablation**
   - train on one dataset, test on others
   - train on subset of model families, test on held-out model family

5. **Information ablation**
   - with oracle fairness statistics
   - with delayed-label estimates
   - with strictly causal online signals only

6. **Action-space ablation**
   - binary intervene/not-intervene
   - three-action keep/flip+/flip-

## 11. What you need to build next

### High-priority technical work

1. Replace placeholder online state components with a principled estimator.

Options:

- delayed-label rolling estimates,
- exponentially weighted estimates once labels arrive,
- causal/forecasted proxies,
- or drop unavailable features entirely and retrain.

2. Clean the reproducibility pipeline.

- pin dependency versions,
- provide one-command training and evaluation,
- re-save all artifacts in the final environment,
- avoid silent fallback behavior when the RL model fails to load.

3. Separate paper evaluation from demo logic.

- no hardcoded demo cases in experimental scripts,
- separate benchmark code path from UI/demo path.

4. Add robust baselines and evaluation scripts.

5. Add statistical confidence.

- multiple seeds,
- mean and standard deviation,
- paired significance tests where appropriate.

### Medium-priority work

6. Expand fairness definitions.

Right now the story is strongest for demographic parity style control. Equal opportunity and equalized odds would make the paper stronger if the online estimation problem is handled carefully.

7. Add delayed-feedback realism.

Many deployment settings do not have immediate labels. A paper that explicitly handles delayed feedback would be more novel.

8. Add a constrained or calibrated policy variant.

Your current training metrics suggest occasional overcorrection and instability. That opens a good second contribution:

**constraint-aware or calibration-aware fairness control**.

## 12. The paper I would write from this repo

### Recommended paper outline

1. Introduction
2. Related work
   - fair RL
   - fairness under drift
   - post-processing fairness
   - runtime fairness monitoring
3. Problem setup
   - deployed black-box classifier
   - sequential stream
   - protected groups
   - fairness target
4. FairFlow
   - universal state
   - action space
   - reward
   - training on synthetic scenarios
5. Experimental setup
   - datasets
   - models
   - baselines
   - metrics
   - drift protocols
6. Results
   - overall transfer
   - fairness/utility tradeoff
   - intervention efficiency
   - stability under drift
7. Ablations
8. Limitations and ethics
9. Conclusion

### Central thesis sentence

**A fairness controller can be learned once, attached to many black-box classifiers, and used as a low-touch runtime guardrail for fairness under deployment shifts.**

## 13. Is this publishable?

### Short answer

Yes, but not yet in its current form.

### More honest answer

The idea is publishable.

The current repository is a strong hackathon prototype plus an early research prototype.

To become a credible conference paper, you still need:

- a tighter formal problem statement,
- a cleaner deployment-time information model,
- rigorous baselines,
- stronger experiments under shift,
- reproducibility cleanup,
- and honest limitation reporting.

### Estimated work

If execution is focused, this is roughly:

- **2-4 weeks** to clean the system and build a reproducible evaluation stack,
- **2-4 weeks** to run experiments, ablations, and multiple seeds,
- **2-3 weeks** to write, revise, and polish the paper.

Realistically: **6-10 weeks** for a solid conference-ready submission.

## 14. Best venue strategy still open on 2026-04-28

These are the most realistic currently open options I found from official pages.

### 1. IEEE IRAI 2026

- Official site: https://irai2026.ieee-ies.org/
- Conference dates: **September 3-5, 2026**
- Extended paper deadline: **April 30, 2026**
- Proceedings: accepted and presented papers published in **IEEE Xplore**
- Fit: very strong on fairness, responsibility, governance, deployment

Assessment:

- Best thematic fit among the currently open venues.
- The deadline is effectively immediate, so only realistic if you submit an early version or extended abstract-style full paper very soon.

### 2. SGAI AI-2026

- Official CFP: https://www.bcs-sgai.org/ai2026/?section=call
- Conference dates: **December 15-17, 2026**
- Submission deadline: **June 26, 2026**
- Proceedings: **Springer LNAI / LNCS**
- Fit: good practical AI venue, friendly to applied system papers

Assessment:

- Best balance of time and realism.
- If you need 6-10 weeks to turn this into a strong paper, this is the safest target.

### 3. IEEE ICTAI 2026

- Official site: https://ictai.computer.org/2026/
- Paper deadline: **June 30, 2026**
- Notification: **September 10, 2026**
- Proceedings: **IEEE Computer Society**
- Fit: broad AI venue, less fairness-specific, but technically appropriate for the RL/control and transfer angle

Assessment:

- Good backup if you frame the paper as a technical AI systems/control contribution rather than primarily ethics/policy.

### 4. IEEE BigData 2026

- Official site: https://bigdataieee.org/BigData2026/
- Conference dates: **December 14-17, 2026**
- CFP page active; check the paper-submission page on the site for the final paper date before committing
- Proceedings: IEEE venue, typically suitable for data-driven deployment and monitoring work
- Fit: good if you position the paper around streaming fairness monitoring/control under drift

Assessment:

- More of a framing-dependent venue than the first three.

## 15. Indexing notes

Be careful here and do not overclaim in submissions.

- IEEE IRAI states accepted papers are published in **IEEE Xplore**.
- IEEE ICTAI states proceedings are published by the **IEEE Computer Society**.
- SGAI AI-2026 states proceedings are published by **Springer LNAI/LNCS**.

For Springer venues, LNCS/LNAI proceedings are generally associated with indexing pipelines such as **Scopus** and **Conference Proceedings Citation Index / Web of Science**, but you should still verify the exact volume once the proceedings record is finalized.

Relevant official links:

- IEEE conference publication into Xplore: https://events.ieee.org/planning-basics/ieee-conference-publications/submitting-proceedings-ieee-xplore/
- Springer support on indexing: https://support.springernature.com/en/support/solutions/articles/6000274014-which-indexes-will-index-my-article-

## 16. Final recommendation

If you want the strongest and most honest paper from this codebase, pursue this direction:

**FairFlow as a universal RL fairness guardrail for deployed black-box classifiers under streaming drift.**

Do not try to sell it as:

- a general fair RL breakthrough,
- a complete compliance platform,
- or a fairness solution for all settings.

Sell it as:

- a post-deployment controller,
- with transfer across datasets and models,
- evaluated under stream-order and drift stress,
- with clear intervention-efficiency tradeoffs.

That is novel enough to matter, close enough to your existing code to be feasible, and aligned with the current research landscape.

