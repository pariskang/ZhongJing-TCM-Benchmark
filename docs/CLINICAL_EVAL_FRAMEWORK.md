# Clinical Evaluation Framework — from "answering" to "clinical decision process"

This document is the **evaluation standard** ZhongJing-TCM commits to. It states
how questions should be shaped and scored to approximate real clinical ability of
LLMs / agents, and maps every requirement to a pipeline module with an honest
**status** (✅ implemented · 🟡 partial · ⬜ planned). It is the design contract;
the generation prompt (`prompts/gen_question.v5.txt`) and the DTQF filter (M6)
already enforce the parts marked ✅/🟡.

> Literature base (2024–2026): AgentClinic (npj Digit. Med. 2026), MediQ
> (NeurIPS 2024), HealthBench (OpenAI 2025), MAQuE / LLM-Mini-CEX, MedR-Bench /
> MedAgentsBench, the process-reward line (PRM800K → AgentProcessBench /
> ToolPRMBench / Plan-RewardBench / Med-TIV), τ-Bench / τ²-Bench.

## 0. Why static MCQ has a validity cliff

Static MCQs assume three things that are false in real clinics: (1) information
is complete, (2) the decision is a single point, (3) you only "say", never "do".
Reframing MedQA as a sequential encounter where the agent must *reveal* the
diagnosis collapses accuracy by ~10× (AgentClinic). So fidelity must climb from
"complete-information single choice" toward "sequential decision under partial
observability".

## 1. Unifying view — the encounter as a POMDP

| POMDP element | Clinical meaning | TCM analogue |
|---|---|---|
| hidden state `s` | true pathology (unobserved) | true 证型 / 病机 |
| observation `o` | complaint, exam, labs, imaging (partial, noisy) | 四诊所得 (望闻问切) |
| action `a` | ask / order test / diagnose / treat / refer / escalate | 进一步四诊 / 辨证 / 立法 / 处方 / 调方 |
| belief `b` | distribution over differentials | candidate-syndrome distribution |
| reward `r` | correctness + efficiency + safety + experience | + 方证契合度 |

**Master rule:** clinical ability = efficiently reducing diagnostic uncertainty
and acting safely under partial observability and risk. Four cross-cutting axes
follow: **information value**, **timely closure** (no premature closure / no
endless probing), **action validity** (no tool-grounding hallucination),
**risk & honesty** (red flags, contraindications, abstain when under-informed).

## 2. Fidelity ladder (T0–T6)

| Tier | Question type | What static MCQ can't test | Status in this repo |
|---|---|---|---|
| **T0** | Static best-answer | knowledge breadth (anti-cheat anchor) | ✅ M1–M7 produce single/multiple/short across 9×3×3 |
| **T1** | Sequential information unlocking | differential under staged info | ✅ `src/t1_counterfactual.py` — counterfactual minimal pairs (flip one 四诊 feature → answer flips; pair-accuracy + flip-rate) and cumulative information-staging (information efficiency); `run.py counterfactual`. v5 prompt also mandates a complete disease course |
| **T2** | Active inquiry (patient simulator) | question quality, info value, timely closure, abstention | ✅ `src/t2_patient_sim.py` — zero-leak `PatientSim`, ask→answer loop, scoring (turns / key-feature recall / premature closure / abstention); `run.py consult` |
| **T3** | Tool-use agent | order test / retrieve / check contraindication; tool-grounding | ✅ `src/t3_tools.py` — deterministic 十八反/十九畏 + dose checkers, a call→result agent loop, and **tool-grounding contradiction detection** (answer vs observed tool result); `run.py tools` |
| **T4** | Longitudinal episode (follow-up) | adjust plan from outcome feedback; trajectory consistency | ✅ `src/t4_longitudinal.py` — outcome-dependent evolution (wrong Tx → 入里化热), per-visit + trajectory scoring (resolution / adverse transitions / adjustment recall / clean resolution); `run.py episode` |
| **T5** | Multi-agent / MDT | collaboration, disagreement resolution, escalation | ✅ `src/t5_mdt.py` — specialty panel, confidence-weighted majority / chair aggregation, **group-vs-individual** (correct vs amplify), disagreement + red-flag recall; `run.py mdt` |
| **T6** | Open rubric dialogue | communication, empathy, safety, completeness | ✅ `src/t6_dialogue.py` — multi-turn transcript graded by a consensus-filtered, axis-tagged rubric (reuses L3/L4), with a hard subset; `run.py dialogue` |

A serious benchmark places items at several tiers; T0 today, T1 strengthened by
v5, T2–T6 are the roadmap below.

## 3. TCM-specific innovations (differentiating value)

TCM epistemology fits the POMDP frame natively:

- **辨证 = active multimodal acquisition.** 四诊合参 is choosing observation
  modalities to disambiguate the syndrome. *(T2/§Roadmap)*
- **同病异治 as a manifold, lifted to trajectories.** ✅ `t4_longitudinal` —
  the syndrome evolves with the treatment chosen (风寒误治入里化热 → 及时转清热),
  scoring whether the plan migrates correctly across visits (`adjustment_recall`,
  `clean_resolution`), not just a single-point valid-set membership.
- **Counterfactual minimal pairs.** Fix the vignette, flip one 四诊 feature
  (舌淡↔舌红) → the correct syndrome/treatment must flip. **v5 already requires a
  decisive discriminating feature with exactly this property** 🟡.
- **Classics as reasoning, not recitation;** anchor textbook version to avoid
  answer drift (M4 anchors / §6.4 of the static standard).

## 4. Four orthogonal scoring layers

Report each layer independently; never collapse to one accuracy number.

| Layer | Scores | Key metrics | Status |
|---|---|---|---|
| **L1 Result** | final dx / syndrome / treatment | single-key correct / manifold membership | ✅ M8 accuracy·P·R·F1; short-answer semantic judge |
| **L2 Process** | reasoning & action trajectory | step-PRM: info efficiency, no premature closure, grounded chain | ✅ `src/l2_process.py` — step-PRM cases (correct / plausible-wrong / neutral), process-preference accuracy, and a result/process gate (premature-closure correct → downgraded); `run.py process` |
| **L3 Safety** | harm avoidance | red-flag detection & escalation, contraindication/dose, hedging vs over-confidence | ✅ `src/l3l4_rubric.py` — weighted safety axis with **negative** items (contraindications); M8 refusal/abstention primitive; v5 bakes safety into items |
| **L4 Interaction** | communication & experience | clarity, empathy, anti-sycophancy | ✅ `src/l3l4_rubric.py` — communication / context-seeking / hedging axes; `run.py rubric` |

**Iron law — decouple result from process:** "right answer, wrong reason"
(guessing / shortcut / position bias) passes L1 but must fail L2 (a *process
gate*); "wrong final, sound process" earns partial credit.

## 5. Scientific-measurement controls

These turn the benchmark from a demo into an instrument; most TCM benchmarks are
weakest here.

- **Patient-simulator validity (T2–T5):** consistency, **zero answer/label
  leakage** (symptom-level facts only — never the syndrome name), realism
  (human reader study), persona diversity (MAQuE), non-collusion (simulator ≠
  graded model). ⬜
- **Judge reliability:** meta-evaluate the grader against physician labels
  (κ/concordance). ✅ `l3l4_rubric.meta_evaluate` (the demo keyword judge scores
  κ≈0.83 vs physician labels — i.e. *not* 1.0, which is exactly why judges must be
  validated). Heterogeneous / tool-grounded judges to mitigate shared blind spots
  ✅ `src/judges.py` — a tool-grounded judge catches a 附子+半夏 contraindication the
  keyword judge misses; `judge_agreement` flags same-source spurious agreement;
  `run.py judges`.
- **Robustness battery:** **option-order & label-symbol invariance** ✅
  (`m8_evaluate.evaluate_invariance` / `run.py invariance` — shuffle + A–D↔甲乙丙丁
  /1–4, reports accuracy drop & content-level consistency); bias injection +
  fairness gap, 四诊/lab noise, paraphrase invariance, sycophancy probes ⬜.
- **Abstention calibration:** A@D, premature-closure rate, missing-premise
  abstention. ✅ `src/abstention.py` — reuses the M8 refusal detector to score
  abstention precision/recall (= A@D)/F1 on missing-premise items + the
  over-abstention rate on answerable ones; `run.py abstain`.
- **Confidence calibration:** ✅ `src/calibration.py` — ECE / Brier / reliability
  bins from elicited answer+confidence (the over-confident mock scores ECE≈0.4);
  `run.py calibrate`.
- **Contamination:** new-vs-old case performance gap, private held-out, dynamic
  injection. 🟡 (synthetic-only release + MinHash de-dup already defend leakage)
- **Dual signal:** always collect end-to-end **and** step-level signals. ⬜

## 6. Capability × tier matrix (blueprint)

| Capability ＼ tier | T0 | T2 | T3 | T4 | T6 |
|---|---|---|---|---|---|
| 辨证/diagnosis | ● | ●(四诊采集) | ●(order test) | ●(syndrome evolution) | ○ |
| 立法/处方 | ● | ○ | ●(contra-check) | ●(adjust) | ○ |
| safety/red-flag | ○ | ●(escalate) | ●(dose/compat) | ●(adverse) | ●(emergency) |
| communication | — | ●(MAQuE persona) | — | ●(adherence) | ●(rubric) |
| classics/principle | ●(derive) | ○ | ○ | ○ | ○ |

(● primary ○ optional — n/a)

## 7. Admission checklist (every interactive/agent item)

A. Tier & capability named; tests something static MCQ can't.
B. POMDP-legal: initially under-determined **but** a path reaches a unique
   correct terminal (or declared multi-answer→manifold); reveal order doesn't
   leak the answer; ≥1 tool-result-vs-claim contradiction point (T3+).
C. Simulator (T2+): consistency pass, zero dx/syndrome leakage, non-collusion,
   diverse personas, (ideally) human realism check.
D. Four layers wired: L1 rule; L2 step cases ("correct action + plausible wrong
   action" + neutral negatives); L3 safety items weighted; L4 multi-physician
   consensus rubric; result/process reported decoupled.
E. TCM: 四诊 info-gain measurable & missing-discriminator penalised; 同病异治 on
   trajectory; counterfactual minimal pair; classics test reasoning, version
   anchored.
F. Controls: judge meta-eval, perturbation battery, abstention calibration,
   contamination checks, dual signal collected.

## Roadmap (phased, mapped to modules)

- [x] **Order/symbol-invariance perturbations (M8).** Shuffle options / relabel
  A–D↔甲乙丙丁/1–4, report accuracy drop + content-level consistency.
  `m8_evaluate.evaluate_invariance`, `run.py invariance`.
- [x] **T2 patient simulator (`src/t2_patient_sim.py`).** Zero-leak `PatientSim`
  over the existing `LLMClient`, ask→answer loop, and an inquiry-efficiency /
  timely-closure / premature-closure / abstention scorer. `run.py consult`.
- [x] **T1 counterfactual pairs + information staging (`src/t1_counterfactual.py`).**
  Flip one 四诊 feature → answer flips (pair-accuracy + flip-rate); cumulative
  staging → information efficiency. `run.py counterfactual`.
- [x] **L2 process gate + step-PRM data (`src/l2_process.py`).** Step cases with
  correct / plausible-wrong / neutral actions; process-preference accuracy; a
  result/process gate that downgrades premature-closure correct answers.
- [x] **L3/L4 rubrics + judge meta-evaluation (`src/l3l4_rubric.py`).** Weighted,
  axis-tagged, positive/negative rubric items; Cohen's-κ meta-eval vs physician
  labels.
- [x] **Abstention probes (`src/abstention.py`).** Missing-premise items scored
  with the M8 refusal detector → A@D (recall) + over-abstention rate. `run.py abstain`.
- [x] **T3 tool-use agent (`src/t3_tools.py`).** 十八反/十九畏 + dose checkers, a
  call→result loop, and tool-grounding contradiction detection. `run.py tools`.
- [x] **T4 longitudinal episode (`src/t4_longitudinal.py`).** Outcome-dependent
  syndrome evolution; trajectory scoring (resolution / adverse transitions /
  adjustment recall / clean resolution). `run.py episode`.
- [x] **T5 MDT multi-agent (`src/t5_mdt.py`).** Specialty panel + aggregation,
  group-vs-individual (corrected vs amplified), disagreement & red-flag recall.
  `run.py mdt`.
- [x] **T6 open rubric dialogue (`src/t6_dialogue.py`).** Multi-turn transcript ×
  consensus-filtered axis rubric + hard subset. `run.py dialogue`.
- [x] **Confidence calibration (`src/calibration.py`).** ECE / Brier / reliability
  bins. `run.py calibrate`.
- [x] **Heterogeneous / tool-grounded judges (`src/judges.py`).** Tool-grounded
  override + ensemble + judge-vs-judge agreement. `run.py judges`.

**The T0–T6 ladder and all four scoring layers, plus the measurement controls
(invariance, abstention, calibration, judge meta-eval & tool-grounding), are now
implemented.** Remaining extensions: real physician-authored rubrics & patient
simulators, private held-out sets, and richer tool environments.

Contributions are welcome against any roadmap item; open an issue referencing
the tier/layer and the admission checklist above.
