# LLM Quality Eval Results

This directory holds run artifacts from `run_recommendation_judge.py` and `run_pairwise_baseline.py`.

Layout per run:

```
runs/<timestamp>_<label>/
  config.json                          # exact CLI args, model versions, persona file hash, rubric file hash
  sessions/<persona>_<query>_seed<N>.json
                                       # input context + raw Recommendation list returned by the system
  judgments/<session_id>_rec<idx>_<judge>.json
                                       # one file per (recommendation, judge, replicate) judgment
  summary.json                         # aggregated metrics: per-criterion means, per-system means, critical-failure counts
```

All raw outputs are gitignored. Only this `README.md` and `.gitignore` are checked in. The committed paper artifacts live in `docs/` (HTML pages with charts).

To resume an interrupted run, pass `--out <existing_run_name>` and the script will skip sessions and judgments whose output files already exist.
