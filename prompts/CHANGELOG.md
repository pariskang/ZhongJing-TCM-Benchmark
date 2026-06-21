# Prompt Changelog

Prompts are **versioned** so the human-in-the-loop refinement loop (manual §5.2)
stays auditable: physicians spot-check a sample of generated questions, tag the
failure modes (ambiguous / out-of-scope / wrong answer / incomplete clinical
presentation), and the prompt is revised accordingly.

## gen_question

### v3 (current)
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
