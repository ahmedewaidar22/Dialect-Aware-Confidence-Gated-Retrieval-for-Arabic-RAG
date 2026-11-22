# Scripts for the reviewer comments (run these yourself; nothing here calls an API)

Copy the four .py files into `ar-route/scripts/`.

1. Freeze before the confirmatory test
   python scripts/freeze_manifest.py --out freeze_manifest.json --files models/dialect_head runs_v2/v2_config.json arroute/pipeline/strategies.py arroute/egy/rewrite.py arroute/pipeline/final.py data/corpus --param k=2 --param lo=0.2 --param hi=0.8 --param n_cands=20 --param generator=gpt-4.1-mini --param generator_snapshot=<dated id from the API response> --param temperature=0
   Commit the manifest, THEN collect/run the new questions. After the run:  python scripts/freeze_manifest.py --verify freeze_manifest.json

2. Human validation of the judge
   python scripts/make_annotation_sheet.py --input <graded answers .jsonl/.csv> --id-field ... --question-field ... --reference-field ... --answer-field ... --judge-field ... --set-field ... --n 300 --out annotation
   Give annotation_A.csv and annotation_B.csv to two annotators (fill the `label` column 1/0). Keep annotation_key.csv private.
   Merge into one CSV with columns: id, ann1, ann2, judge, adjudicated (only where they disagree), set, question, reference, answer.
   python scripts/judge_agreement.py --csv merged.csv --out agreement_report
   -> put the numbers in Section 3.6 and delete its "Status" paragraph.

3. Latency and total cost
   See the docstring of latency_profile.py (wrap the objects load_final() builds, run >= 200 questions with the cache off).
   python scripts/latency_profile.py --selftest   # checks the timing logic only
   -> put the numbers in Appendix D and delete its "Status" paragraph.

Tested here on throwaway synthetic data only (no real results were produced or used).

## Judge validation across all systems (added)

`make_multisystem_sheet.py` and `ranking_effect.py` extend the 300-answer check so that every system is graded on the
same sampled questions and the effect of judge errors on the system ranking is measured.

1. Export one row per (question, system): `qid, system, question, reference, answer, judge, set`.
2. `python make_multisystem_sheet.py --input graded_long.csv --systems "standard_rag,always_c2,adaptive_rag,router_v1_quality,final" --n-questions 100 --out annotation_multi --dry-run` prints how many UNIQUE answers must be graded (identical answers to a question are shown once). Adjust `--n-questions` until the number is workable (about 300-600), then run again without `--dry-run`.
3. Give `annotation_multi_A.csv` and `_B.csv` to the two annotators (blind to judge and system). Keep `annotation_multi_key.csv` private.
4. Disagreements between annotators go in `adjudicated.csv` (`item,label`) after discussion.
5. `python ranking_effect.py --key annotation_multi_key.csv --a annotation_multi_A.csv --b annotation_multi_B.csv --adjudicated adjudicated.csv --reference final --out ranking_effect` prints kappa and PABAK, judge-vs-human accuracy and false-accept/false-reject per system, the ranking comparison (Kendall tau, flipped pairs), the final system against every other system under both label sets, and a test of whether the judge's bias differs between systems.

Both scripts were tested only on synthetic throwaway data. The sample over-represents questions where systems disagree, so
sample accuracies are not the paper's population numbers: report the judge-minus-human gap and whether each conclusion
survives, not the sample accuracy itself.
