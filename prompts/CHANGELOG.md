# Prompt Changelog

Prompts are **versioned** so the human-in-the-loop refinement loop (manual §5.2)
stays auditable: physicians spot-check a sample of generated questions, tag the
failure modes (ambiguous / out-of-scope / wrong answer / incomplete clinical
presentation), and the prompt is revised accordingly.

## gen_question

### v5 (current)
- **Simplified Chinese as the default output language.** A top-priority
  `【输出语言】` block requires every natural-language field (stem, options,
  reference_answer, explanation, theoretical_basis) to be written in
  `{language}` — **defaulting to 简体中文** — even when the source passage is in
  another language; only proper nouns (formulae/herbs/acupoints/classics) may keep
  their Chinese originals, and JSON keys stay English. The language is injected
  from `generate.language` in `configs/pipeline.yaml`, so the default is a real,
  overridable setting rather than an implicit assumption. (v3/v4 carry the same
  directive; the interaction/judge prompts state 简体中文 statically.)
- **Complete disease course for case stems.** Clinical/case stems must now lay
  out the full 发病经过 — sex·age, chief complaint·course (onset, trigger),
  evolution (aggravating/relieving factors, prior treatment & response), current
  symptom cluster, **tongue and pulse (mandatory)**, and relevant history — with
  enough information that the differentiation has a *unique* solution. Mirrors the
  evaluation framework's "complete information / sequential-reveal" principle
  (see `docs/CLINICAL_EVAL_FRAMEWORK.md`, tiers T1/T4).
- **Hard-to-separate options.** Distractors must be *highly confusable* near
  differentials that cannot be told apart by a single symptom or surface
  impression — separable only by integrating several cues with rigorous logic.
- **Decisive discriminating feature (counterfactual-sensitive).** The stem must
  embed one key feature (a tongue/pulse sign or accompanying symptom) such that
  flipping it flips the correct answer, and removing it makes the item
  multi-answer — guaranteeing the options are not trivially separable.
- **Safety awareness** (十八反/十九畏, pregnancy contraindications, toxic-dose,
  red-flag referral) is to be exercised as a correct/distractor consideration.

### v4
- **Self-contained, complete stems.** The stem must be answerable from itself
  alone (no "根据上文/源文本" references, no dangling premises); clinical/case
  stems must spell out sex·age, chief complaint·duration, core symptoms, tongue
  and pulse, plus relevant history — so the syndrome differentiation has a
  sufficient *and unique* basis (kills the "信息不全 / multi-answer" failure mode).
- **High-discrimination options.** Options must be homogeneous and parallel
  (all syndromes, or all formulas, or all therapies/acupoints), similar length,
  no "以上都对/都不对"; distractors are *near-miss* plausible errors drawn from
  confusable syndromes / formulas / common clinical pitfalls, wrong only on
  careful analysis, with no surface cues that give away the key.
- Difficulty-matched stem construction (basic direct / intermediate reasoning
  chain / advanced multi-step clinical scenario).
- `explanation` must now justify *why each distractor is wrong*; short-answer
  `reference_answer` must list complete, itemised scoring points.

### v3
- Added the explicit rule: *clinical questions must contain a complete patient
  symptom/sign description* — directly targets the DTQF completeness filter (M6).
- Required strictly source-grounded answers (no outside knowledge).
- Required distractors that are plausible but unambiguously wrong.
- Switched to fenced-free pure-JSON output with stepwise `explanation` and
  `theoretical_basis` fields.

### v2
- Introduced the three difficulty rubrics (basic / intermediate / advanced).
- Added `reference_answer` for short-answer items.

### v1
- Initial single-/multiple-choice + short-answer generation across 3 difficulties.

## stager_eval
### v1 (current)
- STAGER structured-answer template (paper Table `tcm_prompt`): `[Answer]` block
  + stepwise `[Analysis]` (理论依据 / 关键要点 / 常见误区).

## judge_quality
### v1 (current)
- Three-dimension 0–10 rubric (Professionalism / Popularization / Practicality)
  reproducing the expert radar chart (paper Figure 3).

## judge_validity
### v1 (current)
- Optional LLM completeness/ambiguity check used inside the DTQF review function.
