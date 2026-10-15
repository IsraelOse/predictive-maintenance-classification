# Self-audit

**DAMO510 Take-Home Assignment 2** — Israel Obokohwo, NF1041893, seed 93
Audit performed after `report/DAMO510_A2_Report.docx` was generated, and acted
on before submission. Section 6 records what changed as a result.

Most of the checks below are executable rather than assertions:

| Check | Script | Result |
|---|---|---|
| Fabrication | `src/fabrication_check.py` | 42 of 42 claims verified, 0 mismatches |
| Voice, length, cross-references | `src/verify_report.py` | CLEAN |
| Body page count | `src/page_count.py` | 6 pages, PASS |
| Exact reproduction at seed 93 | `src/reproducibility_check.py` | MATCHES reference |
| Notebook against script pipeline | `src/notebook_parity_check.py` | PARITY CONFIRMED |

The analysis is delivered as one notebook,
`notebooks/DAMO510_A2_full_analysis.ipynb`. It was built from the twelve
scripts in `src/` by `build_notebook.py`, which lifts the code out using the
Python tokenizer rather than retyping it. Section 7 below records the two real
defects that consolidation exposed.

---

## 1. Rubric scoring, argued against myself

I have assumed throughout that the marker is looking for a reason to drop me
from Excellent to Good, and I have written the counter-argument out in full
rather than pretending it does not exist.

### Python Code (CLO4, CLO5) — I judge this **Excellent**

**Evidence for.** All four required models are implemented and tuned, not
defaulted: 316 configurations across 1,580 model fits, with the complete
`cv_results_` tables written to `outputs/tables/cv_{lr,dt,rf,xgb}_full_results.csv`
so the exploration is visible rather than claimed. The twelve stages run in
order from a clean checkout. Preprocessing lives in `ColumnTransformer` objects
built by one shared factory in `src/pipelines.py`, so the baseline, the search,
the imbalance study, the evaluation and the robustness check provably fit the
same transformations. SMOTE is inside an `imblearn.Pipeline`, so it is refitted
per fold and never touches a validation fold.

**The case against, and my answer.** A marker could say `src/pipelines.py` is
not in the required file list. It is an addition, not a substitution: every
numbered script the brief asks for exists and runs. The alternative was four
near-identical copies of the model factory that could silently drift apart, and
I would rather defend one shared module than that.

A second objection: the tuning is randomised for the two ensembles rather than
exhaustive. The spaces are 1,800 and 116,640 combinations, so exhaustive search
is not on the table, and 80 draws of a large space beats a full sweep of an
artificially small one. This is stated in the code and in Section 5 of the
report.

**Residual risk.** The `pywin32` dependency in `src/page_count.py` is
Windows-only. It is a verification tool, not part of the analysis, and nothing
in `src/01` to `src/12` imports it. Documented in `README.md`.

### Cleaned & Transformed Dataset (CLO4) — **Excellent**

**Evidence for.** `data/processed/ai4i2020_model_ready.csv` is produced from
`data/raw/ai4i2020_raw.csv` by `src/03_preprocessing.py`, which prints its
selection, its validity checks and its split before writing. Every
transformation is justified in Table B17, which separates the four steps common
to all models from the one that is model-specific. Scaling is applied to
logistic regression only, and the reason is given rather than assumed.

**The case against.** The processed file contains no scaled columns, because
scaling happens inside the pipeline at fit time. Someone skimming the CSV might
call that "missing key transformations". My answer is that writing a
pre-scaled file would be the leak: the scaler must be fitted on training folds,
so a scaled artifact on disk is either wrong or fitted on everything. Stage 03
prints the fitted design matrix shape and its near-zero post-scaling training
means as proof the step exists.

**Residual risk.** The engineered features are computed before the split. They
are row-wise functions of a row's own sensors, with no cross-row statistic and
no target, so they cannot leak. Stage 03 says so explicitly because it is the
one place a reader might reasonably suspect otherwise.

### Written Report (CLO1, CLO2) — I judge this **Excellent**, with one soft spot

**Evidence for.** All twelve required sections present, body exactly 6 pages by
Word's own pagination. The reflection the rubric asks for is concrete rather
than ritual: Section 7 states plainly that the cost-optimal threshold is
unstable under the assumed ratio, moving from 0.32 at 5:1 to 0.02 at 20:1, and
Section 6.1 reports an ablation result that *weakens* the recommendation
(without the engineered features XGBoost overtakes the Random Forest at 0.8252
against 0.7859). Section 10 gives four limitations, two of them specific to
findings in this analysis rather than generic.

**The strongest case against.** I recommend the Random Forest even though the
Decision Tree beats it on accuracy, precision, recall and F1 at the 0.5
threshold. That is four of six metrics. A marker could reasonably call this
perverse. My answer is in Section 6 and is evidenced: the tuned tree emits only
10 distinct probabilities on the test set against 1,920 for the forest, so it
offers 10 operating points, and the entire recommendation depends on being able
to move the threshold. I think the argument holds, and I accept it is the most
contestable decision in the submission.

**Second objection.** The cost ratio of 10:1 is invented. Stated as an
assumption in Section 7, tested for sensitivity in Table B38, and listed as a
limitation in Section 10. I would rather have a stated assumption with a
sensitivity analysis than an unstated one.

### Artifacts (CLO2) — **Excellent**

25 figures at 300 dpi and 62 tables, every one captioned with what it shows and
why it matters, and every one referenced from the body. `verify_report.py`
enforces this: it fails the build if any figure is never cited or any reference
points at a table that does not exist. Current state is 25 of 25 figures and 62
of 62 tables cited.

**The case against.** The brief required all visuals in the appendix, which
risks a body that reads as disconnected prose. I resolved it by writing every
finding so the sentence carries the number and the figure carries the evidence,
for example "115 records met both conditions and all 115 failed (Table B11)".
A reader can follow the whole argument without turning to the appendix.

### Storytelling & Communication (CLO1, CLO6) — I judge this **Excellent**

**Evidence for.** The report has one spine rather than twelve disconnected task
answers. The exploratory finding in Section 3, that failure risk is U-shaped in
mechanical power, predicts the Section 6 result that logistic regression comes
last, and Section 8 explains the same fact a third way through the wear-torque
importance disagreement. The Section 2 audit finds nine records labelled as
failures with no failure mode set, and three of those exact records reappear in
Section 7 as false negatives. That is a thread, not a summary.

**The case against.** Sections 5 and 6 are dense with numbers. Deliberate, since
the rubric wants results-driven discussion and the brief warned against
algorithm definitions, but a marker who wants more narrative air could mark it
down. Every dense paragraph ends on what the number means rather than on the
number.

---

## 2. Task coverage matrix

| # | Task | Report section | Script | Key outputs |
|---|---|---|---|---|
| 1 | Problem definition | §1 | — | Target, users, FP/FN asymmetry, metric emphasis all present |
| 2 | Data audit | §2, §2.1 | `01_data_audit.py` | B1 to B6, A1 |
| 3 | Exploratory investigation | §3 | `02_eda.py` | B7 to B15, A2 to A8. Six findings, four multivariate |
| 4 | Preprocessing design | §4 | `03_preprocessing.py` | B16, B17 |
| 5 | Feature development | §4.1 | `04_feature_engineering.py` | B18 to B23, A9, A10 |
| 6 | Baseline performance | §5 | `05_baseline.py` | B24, A11 |
| 7 | Required models and tuning | §5 | `06_models_tuning.py` | B25 to B28, B57 to B60, A12 |
| 8 | Class-imbalance investigation | §5.1 | `07_imbalance_study.py` | B29 to B31, A13, A14 |
| 9 | Model evaluation | §6 | `08_evaluation.py` | B32 to B35, A15 to A17 |
| 10 | Threshold investigation | §7 | `09_threshold_analysis.py` | B36 to B39, B61, B62, A18, A19 |
| 11 | Misclassification analysis | §7.1 | `10_misclassification.py` | B40 to B45, A20, A21 |
| 12 | Model interpretation | §8 | `11_interpretation.py` | B46 to B51, A22 to A24 |
| 13 | Final model recommendation | §9 | — | All eight required aspects addressed |
| 14 | Robustness check | §6.1 | `12_robustness.py` | B52 to B56, A25 |
| 15 | Limitations and improvement | §10 | — | 4 limitations (2 specific), 3 improvements |
| — | Analytical Decision Record | §12 | — | 12 rows against a required minimum of 8 |
| — | Assigned seed stated | Title page, §4, §5 | `config.py` | Seed 93 appears 12 times |

**Partial coverage, flagged honestly.**

- Task 13 asks for "model complexity" as an explicit consideration. Section 9
  covers it through the leaves-and-operating-points argument and the XGBoost
  hyperparameter-count remark, but does not give a formal complexity measure
  such as node count or inference latency. I judged that a formal measure would
  cost more space than it earns at 8,000 training rows.
- Task 3 requires "at least two" multivariate findings. I report four. The
  univariate weakness finding is genuinely univariate and is presented as
  scene-setting for the rest rather than counted toward the two.

Everything else is fully covered.

---

## 3. Fabrication check

`src/fabrication_check.py` re-opens the finished DOCX, pulls specific numeric
claims out of the rendered text, and compares each against the JSON or CSV that
produced it. A claim passes only if the exact string appears in the document AND
matches the source value. This catches a builder bug that printed the wrong
number, which a code review of the builder alone would not.

**Result: 42 of 42 claims verified, 0 mismatches.** Sample of what is covered:

| Claim | Source | Value |
|---|---|---|
| Dataset size | `audit.json` | 10,000 rows |
| Failure count and rate | `audit.json` | 339, 3.39% |
| Leakage OR agreement | `audit.json` | 9,991 of 10,000, one-sided (0 the other way) |
| RNF behaviour | `audit.json` | fires 19 times, 1 failure |
| Product ID / Type agreement | `audit.json` | 100% |
| Split | `split.json` | 8,000 / 2,000, 271 training failures |
| Torque univariate AUC | `tbl07` | 0.7702 |
| Heat interaction | `tbl11` | 115 records, all failures |
| Feature gain | `features.json` | 0.7381 to 0.8847 |
| Power-band baseline F1 | `tbl24` | 0.4889 |
| Configurations, fits | `tuning.json` | 316, 1,580 |
| Decision Tree tuning gain | `tbl28` | 0.2061 |
| RF test PR-AUC | `tbl32` | 0.9048 |
| Probability granularity | `tbl35` | 10 vs 1,920 |
| Out-of-fold threshold | `threshold.json` | 0.23 |
| Test-set optimum, for contrast | `threshold.json` | 0.14 |
| Alarms / TP at threshold | `threshold.json` | 63, 57 |
| Regret vs default | `threshold.json` | 110.6% vs 184.5% |
| FN max probability | `misclassification.json` | 0.029 |
| Unflagged overlap UDIs | `misclassification.json` | 1438, 8507, 5537 |
| Spearman DT vs RF | `tbl50` | 0.939 |
| Robustness wins, mean, sd | `robustness.json` | 10/10, 0.9034, 0.0135 |
| Ablation winner | `robustness.json` | XGBoost at 0.8252 |

One mismatch was found on the first run and fixed: the check searched for a
phrasing of the regret sentence that had been superseded when Section 7 was
rewritten. The number itself was correct in both. The check now also verifies
the out-of-fold threshold, the test-set optimum and the recommended threshold's
regret.

---

## 4. Voice check

`src/verify_report.py` scans the rendered document. Current result: **CLEAN**.

- **Em-dashes: 0. En-dashes: 0.** Enforced twice: `build_report.py` asserts on
  both characters at build time, so a report containing one cannot be written,
  and the verifier re-checks the rendered file.
- **Banned phrases: none.** Checked for "it's worth noting", "delve",
  "leverage", "crucial", "pivotal", "landscape", "underscores", "showcases",
  "in today's world", "this analysis demonstrates that", "furthermore",
  "moreover", "in conclusion", "firstly", "secondly", "seamless",
  "cutting-edge", "game-changer", "tapestry", "realm", "testament", "navigate
  the", "dive into", "unlock", "harness", plus a regex for the "not X, but Y"
  construction and one for "robust" used as filler. The word "robustness"
  survives only as the technical name of the Task 14 check.
- **First person for judgements.** "I kept them", "I dropped both", "I retained
  no treatment", "I chose against it", "I am reporting it rather than burying
  it".
- **Uncertainty admitted where it exists.** Section 7: "the argmin is genuinely
  unstable" and "the defensible part of the claim is the direction, not the
  decimal place". Section 6.1: "I would not claim it wins on any feature set".
  Section 10: "I expect a substantial drop and cannot size it from this file".
- **No section opens with a summary of itself.** Section 3 opens on a finding,
  Section 6 opens on the result, Section 9 opens on the recommendation.
- **No tricolon adjective lists, and no paragraph opening with "Furthermore" or
  "Moreover".**
- **Bullet lists in the body: zero.** The body is continuous prose. Lists appear
  only in this audit and in the README.

---

## 5. The three weakest parts

**1. The recommendation contradicts four of six headline metrics.** The
Decision Tree beats the Random Forest on accuracy, precision, recall and F1 at
0.5, and I recommend the forest anyway. The argument rests on probability
granularity, 10 operating points against 1,920, and it is evidenced in Table
B35. It is still the decision most likely to be challenged, and if the plant
genuinely intends to fix the threshold and never revisit it, the tree is the
better answer. Section 6 says so.

*A stronger version* would run the threshold analysis on both models and show
directly what the tree cannot reach: the tree's 10 operating points would trace
a coarse step function next to the forest's smooth curve, turning a numeric
argument into a visible one.

**2. The cost ratio is invented, and the recommendation inherits it.** 10:1 is
plausible for a machining line and is not sourced from anything. The sensitivity
analysis shows the cost-minimising threshold moving from 0.32 at 5:1 to 0.02 at
20:1, which is a wide range, and everything downstream of Section 7 depends on
where in that range the truth sits.

*A stronger version* would replace the point assumption with an explicit range
elicited from a maintenance planner, and report a threshold band rather than a
single number. Failing that, it would frame the recommendation as "0.20 to 0.30
depending on inspection capacity" and let the plant choose inside it, which is
closer to how the decision would actually be made.

**3. Nothing here tests transfer.** Both robustness checks reuse the same 10,000
records. A different partition of one dataset is not a different machine, a
different plant or a different year, and AI4I 2020 is synthetic, generated from
documented threshold rules on exactly the variables supplied. That is why an
operator rule with no fitting reaches 0.4889 F1 and why the forest reaches
0.9048 PR-AUC. Neither number should be expected to survive contact with real
sensor data.

*A stronger version* would need data this assignment does not have. Within the
data available, the closest honest approximation would be a grouped split by
product grade, training on L and M and testing on H, which at least forces the
model to generalise across a population it did not see. That would have been a
sharper Task 14 than a seed sweep, and I would choose it if I ran this again.

---

## 6. What the audit changed

The audit found one methodological defect and one code defect, and both were
fixed before submission. Before-and-after below.

### 6.1 The threshold was selected on the test set (methodological, fixed)

**Before.** Stage 09 swept thresholds across the test set, took the
cost-minimising value of **0.14**, and then reported test-set performance at
that threshold. The cut-off had seen the data it was scored on, which makes the
reported recall and precision optimistic. No amount of sensitivity analysis
fixes that, because the optimism is in the selection, not the assumption.

**After.** Stage 09 now derives out-of-fold predictions for the 8,000 training
rows via `cross_val_predict` on the same five folds at seed 93, selects the
cost-minimising threshold on those, and carries it to the test set untouched.
The chosen value is **0.23**.

| Selection basis | Threshold | Test recall | Test precision | FN | FP | Test cost |
|---|---|---|---|---|---|---|
| **Out-of-fold, training only (adopted)** | **0.23** | 0.8382 | 0.9048 | 11 | 6 | 5,800 |
| Test set (optimistic, shown for contrast) | 0.14 | 0.8529 | 0.8406 | 10 | 11 | 5,550 |
| Library default | 0.50 | 0.7794 | 0.9815 | 15 | 1 | 7,550 |

The honest choice gives up 250 units against the test-set optimum, 4.5 per cent
of that optimum's cost, and still captures 87.5 per cent of the gain that was
available over the default (1,750 of 2,000 units). Measuring that gap is itself
a result worth reporting, and Section 7 now does.

Downstream effects: Section 7 was rewritten, the recommendation in Section 9
moved from 58 true positives and 11 false alarms to 57 and 6, the
misclassification analysis in Section 7.1 re-ran at the new threshold (the three
unflagged-failure UDIs are unchanged), the regret table now evaluates 0.23
rather than only 0.14, and two new appendix tables were added, B61 and B62.

A related bug surfaced while making this change: the two new tables were
initially numbered `tbl40` and `tbl41`, which collided with stage 10's existing
`tbl40_test_class_medians` and `tbl41_false_positive_cases`. Both files existed
under distinct suffixes so no data was lost, but the numbering was ambiguous.
Renumbered to `tbl57` and `tbl58`, and appended to the appendix as B61 and B62
so no existing table reference shifted.

### 6.2 Permutation importance computed twice per model (code quality, fixed)

**Before.** Stage 11 called `permutation_importance` once inside the per-feature
loop and again to obtain the row for the categorical `type` column, running the
most expensive computation in the project twice for each of four models, with 30
repeats each.

**After.** One run per model, stored in `perm_runs` and reused for both the
per-feature rows and the categorical row. Identical output, half the work.

### 6.3 Reproducibility defect found earlier and fixed

Stage 12 originally rebuilt the alternative-seed partitions with
`train_test_split`, which returns shuffled index arrays. Because Random Forest
bootstrap draws are taken by row position, the seed-93 run in stage 12 produced
a marginally different forest from stage 08, PR-AUC 0.9030 against 0.9048, for
no reason a reader could see. Indices are now sorted inside `split_for`, so the
seed-93 row of the robustness table reproduces stage 08 exactly. This is the
kind of discrepancy that looks like a typo in a report and is actually a bug.

### 6.4 Three narrative claims that contradicted their own numbers

Caught while reading output rather than by a script, and all three corrected in
the code so the report regenerates correctly:

- **Feature ablation.** The draft said dropping the raw temperatures "costs
  predictive information". It does not: variant I scored 0.8895 against 0.8847.
  The text now says the difference is a fifth of the fold standard deviation and
  therefore not evidence of anything, and gives the real reason for keeping the
  raw readings.
- **Threshold sensitivity.** The draft said the recommendation was "not
  knife-edge" and then printed a range of 0.02 to 0.32, which is knife-edge. Now
  states the instability plainly and adds the regret analysis, which is the
  question that actually matters.
- **Model comparison.** The draft framed the best-model discussion as false
  alarms versus missed failures, but at 0.5 the Decision Tree wins both, so the
  contrast was fictional. Rewritten around probability granularity, which is the
  real distinction and is measurable.

---

## 7. What consolidating into one notebook exposed

The brief for the notebook was to move the code, not rewrite it, and to prove
the numbers had not moved. The proof is `outputs/NOTEBOOK_PARITY_CHECK.md`:
63 of 63 tables and 12 of 12 stored results identical, 24 of 25 figures
byte-identical. Getting there took two fixes, and both were real problems
rather than tidying.

### 7.1 The first run used the wrong Python

The first execution produced visibly different Gini importances and
cross-validated scores. The cause was not the analysis: `jupyter nbconvert`
had defaulted to the base Anaconda kernel, which carries pandas 2.3.3 and
scikit-learn 1.7.2, where the project venv has pandas 3.0.5 and scikit-learn
1.9.0. Different scikit-learn builds different forests, and pandas 2 infers
`object` where pandas 3 infers `str`, which is why `tbl01_variable_dictionary`
disagreed on a dtype label.

Fixed by registering the venv as a kernel inside the project itself
(`.venv/share/jupyter/kernels/damo510-a2`, so nothing outside the project
changed) and executing against it. This is worth stating plainly because it is
invisible: the notebook ran to completion with no errors and produced wrong
numbers. Only comparing against the reference caught it.

### 7.2 A CSV round-trip is not bit-exact, and it mattered

With the kernel fixed, a handful of tables still disagreed. The cause was the
one structural change the consolidation makes: passing frames between sections
in memory instead of through CSVs. Writing `power_W` to CSV and reading it back
moves it by about 1e-12, and `wear_torque` likewise. That is meaningless as a
measurement and is still enough to move a tree's split threshold and change a
Gini importance in the third decimal.

The scripts trained on the values as read back from
`ai4i2020_model_ready.csv`. The notebook was training on the in-memory values,
which are not the same numbers. Fixed by loading the file once, immediately
after section 4 writes it, and holding that in memory for the nine sections
that follow. The file is a submitted deliverable, so the dataset the models see
should be the dataset that was delivered, and the notebook now reads it exactly
once instead of nine times.

### 7.3 One figure differs, and it is not the data

`fig25_robustness.png` differs by 0.17 per cent of its pixels, all inside the
left panel. `sns.stripplot` jitters its points using the global NumPy random
state, which is not in the same place in a notebook process as in a script
process. Verified by rendering the same stripplot under two NumPy seeds, then
the same seed twice. Every plotted value is identical and the underlying tables
match exactly. I left the plotting call alone rather than seeding it, because
that call is the one that produced Figure A25 in the report.

---

## 8. Verification commands

```bat
.venv\Scripts\python src\verify_report.py           REM voice, length, cross-references
.venv\Scripts\python src\fabrication_check.py       REM 42 numeric claims vs source files
.venv\Scripts\python src\page_count.py              REM body page count via Word
.venv\Scripts\python src\reproducibility_check.py   REM every result reproduces at seed 93
.venv\Scripts\python src\notebook_parity_check.py   REM notebook vs the script pipeline
```

All five pass as of the submitted build.
