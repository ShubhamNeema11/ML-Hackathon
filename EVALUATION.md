# Evaluation log (26-27 Sep 2026)

All numbers are the **expected official score** (per-S1 macro F0.5 simulation) on held-out records that no model was trained on,
**report half** = the half no threshold was tuned on (the tuned half is shown where useful). Leaderboard (LB) numbers are real submissions.
Nothing here learns from test data; test-set checks marked *(test, no labels)* only describe model behaviour.

## Leaderboard
| file | pipeline | LB |
|---|---|---|
| output_submitted_0957/ | first submission | 0.957 |
| final_overnight/ | B2 | 0.97889 |
| output_best/ | B2 + no-address specialist (all 309,844 no-address training records, tuned) | **0.98011** |

Held-out estimate for B2 was 0.9869 and for output_best 0.9882: the LB reads about 0.008 lower (mostly France, see below), but the
**gain** carried over almost exactly (held-out +0.0013, LB +0.0012).

## No-address pipeline (used)
| experiment | result | used | log |
|---|---|---|---|
| TF-IDF char-3gram blocking for no-address records | owner recall 83.5% -> 93.8% (held-out), 92-93% (train sample) | yes | blocking_noaddr.py output (chat) |
| specialist, 31k records | whole held-out 0.9869 -> 0.9879 | superseded | - |
| specialist, all 309,844 records | 0.9869 -> 0.9881 (no-address group F0.5 0.7870 -> 0.8329) | yes | logs/quick_fit.log, logs/quick_joint.log |
| tuning (4 settings, chosen on the training-internal early-stop split) | best: lr 0.05, 127 leaves (ES logloss 0.03828); whole held-out 0.9882 | yes | logs/night_tuning.log |
| overfitting check | train / early-stop logloss gap 0.0018-0.0039, no overfitting | - | logs/night_tuning.log |

## Tested and rejected (no gain)
| experiment | result | log |
|---|---|---|
| stacking LGB + XGB + CE meta-model | 0.9876 vs 0.9876 | models/decision_stack.json |
| zero-shot bge-reranker-v2-m3 name + address scores, ranker C | 0.9869 vs B2 0.9869 | logs/night_c_fit.log |
| thresholds re-tuned for the test's ~40% orphan share | 0.9868 -> 0.9870 (noise) | decision_check.py (chat) |
| record-level accept model | +0.0002 (full features), +0.0000 (probabilities only) | accept_model.py (chat) |
| French city / state fix | B2 ignores city / state: 100% corrupted 0.9869 -> 0.9872 | (chat) |

## Pass 2 (sibling support) - used
| | report half | tuned half |
|---|---|---|
| pass 1 (B2 + specialist) on the training mirror, 766,438 held-out owned records | 0.9887 | 0.9886 |
| **pass 2** | **0.9893 (+0.0006)** | 0.9892 |
Early stopping at round 278 (validation logloss 0.02261), precision 0.9975 -> 0.9979, recall 0.9688 -> 0.9697. Log: logs/night_pass2_fit.log.

## France (15% of the test set, 0% of training)
*(test, no labels)*: French records sit in the unsure band 0.05-0.98 three times as often as US / India (14.0% vs 5.0-5.6%).
Each fix corrects a mechanism; the mechanism's cost is measured on labelled US / India held-out data damaged the French way (logs/france_eval.log):

| mechanism | labelled score | cost | fix's expected LB effect (x 15%) | in files |
|---|---|---|---|---|
| clean | 0.9869 | - | - | - |
| legal-form codes unknown | 0.9807 | -0.0062 | ~ +0.0009 | france, france2, france3 |
| French address damage | 0.9864 | -0.0005 | ~ +0.0001 | france2, france3 |
| legal forms inside names / dotted | 0.9864 | -0.0005 | ~ +0.0001 | france3 |
| generic list widened (risk check) | 0.9868 | -0.0001 | neutral | france3 |
| address damage in features AND the embedder's text | 0.9862 | -0.0007 (embedder part -0.0002) | ~ +0.00003 for re-embedding France: not done | - |

*(test, no labels)* effect on French records: address similarity of confident matches median 93 -> 100 (share < 90: 30% -> 1.5%);
unsure French share 14.0% -> 13.1% (france) -> 13.0% (france2). US / India predictions byte-identical. Logs: output_france*/FRANCE_CHECK.txt.

SHAP (france_shap.py, test candidates, no labels): the largest French penalty is the dense channel (dense_margin -0.32, dense_score -0.27 log-odds vs US / India);
clean French text raises French dense cosine 0.915 -> 0.967, but on labelled data the embedder's share of the damage is only -0.0002, so the dense penalty is not a
text-format problem (more likely more look-alike candidates in the French data).

## Bigger models (Stage 1 on AWS, L4 GPU; judged on held-out data)
| model | held-out result | verdict |
|---|---|---|
| embedder BAAI/bge-m3 fine-tuned (600k training pairs) | owner recall dense@10 0.9907 -> 0.9914, union 0.9921 -> 0.9927, dense@1 0.9744 -> 0.9770 | borderline |
| reranker BAAI/bge-reranker-v2-m3 fine-tuned listwise (65,940 groups) | band logloss with B2 p: current CE 0.0649, new reranker alone 0.0659, both 0.0643 (-0.8%) | no gain (zero-shot gave -3.7% and 0.0000 official) |
Stage 2 (new candidates + reranker features, 5-6 h) not run: expected gain too small. Logs: logs/rr_judge.log, stage1 RESULT (chat).

## Root cause: an unseen country (leave-one-country-out, loco.py / loco2.py, labelled data)
Model trained on US only, India held out as the unseen country (as France is for us):
| variant | India (unseen) | US |
|---|---|---|
| trained on US + India (seen) | 0.9874 | 0.9847 |
| US only, B2's features | 0.9586 | 0.9850 |
| US only, country-relative percentile features | 0.9151 | 0.9267 |
| **US only, without country codes (state / legal codes)** | **0.9620** | 0.9851 |
| US only, without codes and without name-length / state / city flags | 0.9598 | - |
| US only, without codes, thresholds tuned on India (oracle calibration) | 0.9628 | - |
An unseen country costs ~0.025-0.03 and it is neither thresholds (+0.0008 at best) nor country features: the model has not learned that
country's record variation. The one generic gain: drop the country codes (+0.0034 on the unseen country, seen country unchanged).
Logs: logs/loco.log, logs/loco2.log.

## Final file
output_final/ = B2 + specialist + all French fixes (france3) + pass 2. Expected LB about **0.982** (0.98011 + ~0.0011 France + ~0.0006 pass 2).

## Raw-name reranker in the name-only pipeline (noaddr_cdrq)

- Reranker `models/ce_nar` (multilingual-e5-small, listwise, raw lowercased names only, training data only) scores the top-8 candidates per
  no-address record (chosen by noaddr_cdr). Features `nar_score / nar_gap / nar_rank` added to the specialist → `models/noaddr_cdrq.txt`
  (lr 0.05, 127 leaves, min_leaf 400, code dropout 0.5; best iteration 811). Logs: `logs/cdrq_fit.log`, `logs/cdrq_joint.log`, `logs/cdrq_compare.log`.
- Specialist alone (held-out eval, orphan weight 0.1): F0.5 0.8402 (cdr) → **0.8470** (cdrq); the reranker features are the top features.
- Joint held-out metric at the SAME rule (addr 0.8/0.3, no-address 0.8/0.3): cdr half A 0.9888 / half B 0.9885 → cdrq **0.9891 / 0.9886**
  (+16 true links, −1 wrong link). cdrq ≥ cdr in 9/9 rule cells on half A and 7/9 on half B; averaging cdr+cdrq is not better than cdrq.
- The joint tuner picked no-address 0.7/0.2 for cdrq (report half 0.9884): lowering the threshold adds more wrong links than it gains → keep 0.8.
- Test side: `run_final_rq.sh` → `output_cd_p2xrq/` (pass 2 + decision_pass2_exact.json).

## LB: output_cd_p2xrq scored BELOW output_final (0.9809)
- output_cd_p2xrq accepts +45k pairs vs output_final; nearly all come from the exact-metric rule (addr 0.65, no-address 0.5), which was
  tuned on the training mirror (26% records without a match) - test has ~40%, so those extra links are wrong far more often on test.
  The generic model with the old rule (output_cd_p2) differs from output_final by only ~17-18k pairs each way.
- New file with the old (orphan-reweighted) rule decision_pass2.json: `output_cd_p2rq/` (generic + raw names + raw-name reranker),
  5,836,988 pairs (output_final 5,831,107; 18k removed, 24k added). Validator PASS. Log: logs/rq_write_oldrule.log.

## Does the address reranker depend on the state code? (logs/cehide_eval.log, crossenc ER_CE_HIDE_STATE)
Held-out eval pairs re-scored by ce_er2 with the trailing state code removed from both texts (no training):
| | reranker AUC (address pairs) | ranker_cd half A | half B |
|---|---|---|---|
| full text | 0.9910 | 0.9875 | 0.9872 |
| state hidden | 0.9907 | 0.9874 | 0.9867 |
Scores move by 0.009 on average: the reranker barely uses the state token, so retraining it with the state hidden (8-10 h) is not
expected to change the unseen-country result. Not pursued.

## Separate rerankers per side with a PRETRAINED reranker (crossenc ER_CE_SIDE / ER_CE_PRETRAINED; logs/bge_compare.log)
BAAI/bge-reranker-v2-m3 (568M, multilingual, off the shelf, no training) vs our fine-tuned rerankers, held-out eval pairs, each side alone.
Name-only side = raw names; address side = normalized name + full address. Records whose owner is among >= 2 candidates:
| side | reranker | AUC | top-1 correct | identical-name twins |
|---|---|---|---|---|
| name-only (1,401 rec) | ours ce_nar (e5-small, fine-tuned) | **0.8876** | **0.7402** | 0.2785 |
| | bge-reranker-v2-m3 (pretrained) | 0.8194 | 0.6809 | 0.2626 |
| | average of both | 0.8751 | 0.7352 | 0.2812 |
| address (4,610 rec) | ours ce_er2 (fine-tuned) | **0.9910** | 0.9894 | **0.9973** |
| | bge-reranker-v2-m3 (pretrained) | 0.8340 | 0.9783 | 0.9753 |
| | average of both | 0.9514 | 0.9902 | 0.9918 |
The pretrained reranker is worse on both sides (it ranks query-passage relevance, not "same business"); averaging adds nothing.
Speed on the laptop GPU: ~1,000 pairs/s name-only, ~420/s address.

## Name-only rerankers pretrained for company names (logs/name_models_compare.log)
Same held-out name-only pairs, raw names (1,401 records with the owner among >= 2 candidates, 377 with an identical-name twin):
| reranker | AUC | top-1 | non-twins | twins |
|---|---|---|---|---|
| ours ce_nar (fine-tuned on our name-only data) | **0.8876** | **0.7402** | **0.9102** | 0.2785 |
| bge-reranker-v2-m3 (general) | 0.8194 | 0.6809 | 0.8350 | 0.2626 |
| Vsevolod/company-names-similarity-sentence-transformer | 0.7784 | 0.6453 | 0.7988 | 0.2281 |
| easonanalytica/cnm-multilingual-small-v2 | 0.6868 | 0.3904 | 0.4658 | 0.1857 |
Company-name models call any similar name a match; all our candidates already have similar names. Ours stays.

## Exact-metric rule with records without an owner weighted to the test share (exact_tune.py ER_TEST_ORPHAN_SHARE; logs/exact_tune_p2_orphan.log)
Correction: no run ever set ER_ORPHAN_SHARE, so decision_pass2.json (output_final's rule) was tuned at the training share 0.26 too.
| test share | old rule (0.8, 0.7, 0.1) A / B | exact-tuned rule A / B |
|---|---|---|
| 0.40 | 0.9877 / 0.9882 | (0.65, 0.5, 0.3, 0.3) 0.9882 / 0.9883 |
| 0.45 | 0.9874 / 0.9880 | (0.65, 0.6, 0.3, 0.3) 0.9879 / 0.9880 |
The lower-threshold rule still wins at the test share, so it is probably NOT why output_cd_p2xrq scored below output_final. The remaining
difference is the country-specific normalization rules (off in the generic files), which offline data cannot measure.
Next: run_cd_rules.sh = generic models + rules back on at test time (no training) -> output_cdr_rules / output_cdr_rules_x.
