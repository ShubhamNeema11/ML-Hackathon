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

## Final file
output_final/ = B2 + specialist + all French fixes (france3) + pass 2. Expected LB about **0.982** (0.98011 + ~0.0011 France + ~0.0006 pass 2).
