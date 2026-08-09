# Phase C — Human Calibration Round 1 (Annotator Instructions)

Thank you for taking part. There are 2 of you (pi, student)
and you are independently rating the same 50 recommendations. Please do not discuss
ratings with each other before all CSVs are returned.

## What you will do

You have been given a CSV (`annotations_<your_name>.csv`) with one row per recommendation.
For each row, fill in the seven blank columns at the right:

  - C1_personalization      (1-5 integer)
  - C2_risk_alignment       (1-5 integer)
  - C3_factual_grounding    (1-5 integer)
  - C4_actionability        (1-5 integer)
  - C5_diversification      (1-5 integer)
  - C6_safety_compliance    (1-5 integer)
  - comment                 (free text, optional, useful when score <= 2)

Use the rubric in `judge_rubric_rec.md` (Section 2). The exact same rubric is being used
by the LLM judges; this calibration round measures how the LLM judges agree with humans.

## Important

- You are blind to which system produced each recommendation. Score by the rubric only.
- Do NOT look at LLM-judge scores before finishing your annotations.
- Score independently. Do not consult the other annotators until all CSVs are returned.
- If you find a row where the rec is malformed (e.g., empty fields), score what is
  present per the rubric and add a comment explaining.

When you are done, return the filled CSV to the calibration coordinator.
