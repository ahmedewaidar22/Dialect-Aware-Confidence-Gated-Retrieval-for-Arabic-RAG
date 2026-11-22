# Confirmatory test of the frozen final system

Put the four .py files in `ar-route\scripts\`. Run everything from `D:\Masters\ar-route-local\ar-route` in PowerShell.
`freeze_manifest.py` is already in your review folder (copy it to `scripts\` or call it from `review\`).

## A. Freeze (before any new question exists)
```powershell
python review\freeze_manifest.py --out freeze_manifest.json `
  --files models\dialect_head runs_v2\v2_config.json arroute\pipeline\strategies.py arroute\pipeline\final.py `
          arroute\pipeline\judge.py arroute\egy\rewrite.py arroute\llm.py arroute\router\v2.py `
          scripts\run_confirmatory.py scripts\analyze_confirmatory.py data\corpus `
  --param k=2 --param lo=0.2 --param hi=0.8 --param n_cands=20 --param generator=gpt-4.1-mini --param temperature=0
git add freeze_manifest.json
git commit -m "Freeze final pipeline before confirmatory test"
```
(if the folder is not a git repo, skip the two git lines and keep a dated copy of `freeze_manifest.json` somewhere safe.)
Write down the date, and the commit id, in the paper.

## B. Draw passages and collect questions
```powershell
python scripts\sample_passages.py --n 600 --seed 20261101 --out data\confirm\passages_to_write.csv
```
Give `passages_to_write.csv` to 2+ native Egyptian speakers (split the rows between them). For each passage they fill
q_egy, q_msa, answer (see the header in the script). Rules for writers: write the question yourself, do not use an AI
tool, do not look at any system's output, skip a passage that gives no clear factual question (leave the row empty).
Save as CSV UTF-8.

## C. Build the question file (needs >= 500 usable rows)
```powershell
python scripts\build_confirm_set.py --sheet data\confirm\passages_to_write.csv --out data\confirm\confirm_test.jsonl
```

## D. Smoke test on 5 questions, then verify nothing changed
```powershell
python scripts\run_confirmatory.py --limit 5 --budget_usd 0.5 --out runs_confirm\smoke.jsonl
```
If it errors, fix only the RUNNER if the error is in the runner itself, and tell me; do not touch any frozen file.
(An error inside a frozen file means the freeze must be redone and the date recorded.) Delete `runs_confirm\smoke.jsonl`.

## E. The real run (about US$1 for 500 Egyptian questions at the paper's prices)
```powershell
python review\freeze_manifest.py --verify freeze_manifest.json
python scripts\run_confirmatory.py --budget_usd 3.0
python review\freeze_manifest.py --verify freeze_manifest.json
```
Both verifies must print FROZEN OK. The run resumes if interrupted: run the same command again.

## F. The pre-specified tests
```powershell
python scripts\analyze_confirmatory.py
```
Paste the whole output to me. It prints PASS or FAIL for H1, H2 and H3 and all the numbers. Report it whatever it shows.

## G. Then grade the new questions' judge verdicts (recommended)
Reuse the annotation scripts on `runs_confirm\results.jsonl` (judge validation for the new run), if time allows.
