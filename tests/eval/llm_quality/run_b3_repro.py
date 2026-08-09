"""A-vs-B3 reproducibility re-run (reviewer point 3).

Reuses run_pairwise_baseline's machinery but restricts generation/judging to the
A_wealthnexus vs B3_generic_llm pair, to test whether the headline external result
(originally 50/5/17, 90.9% decided, p<0.001) reproduces run-to-run under the same
non-deterministic pipeline (generation temp 0.4, etc.) that failed to reproduce A-vs-B2.

Usage:
  python -m tests.eval.llm_quality.run_b3_repro
"""
import sys
from tests.eval.llm_quality import run_pairwise_baseline as rp

# Restrict to A and B3 only (skip B1/B2 generation + judging => big cost saving).
rp.SYSTEMS = ["A_wealthnexus", "B3_generic_llm"]
rp.PAIRS = [("A_wealthnexus", "B3_generic_llm")]

if __name__ == "__main__":
    sys.argv = ["run_pairwise_baseline", "--full", "--out", "20260808_b3_repro_pw"]
    raise SystemExit(rp.main())
