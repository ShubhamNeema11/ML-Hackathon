# Post-mortem: business entity resolution hackathon (ended 2026-09-27)

Final leaderboard score **0.98157**, rank ~950 (all India). Top teams were reported at ~0.989. Details of every experiment are in
EVALUATION.md; this page covers only what mattered.

## Leaderboard history

| file | what changed | LB |
|---|---|---|
| B2 | ranker + cross-encoder | 0.97889 |
| output_best | + separate no-address (name-only) specialist | 0.98011 |
| output_final | + country-specific normalization rules + pass 2 (sibling re-ranking) | 0.9809 |
| output_cd_p2xrq | generic model (code dropout) + raw names + name-only reranker, **lower thresholds** | < 0.9809 |
| output_cd_p2rq | same scores, previous thresholds | 0.9815 |
| output_cd_p2rq_split | lower thresholds only for countries present in training | **0.98157** |

## What actually moved the score

1. **Separate pipeline for records without an address** (+0.0012): the single biggest gain after the base model.
2. **Pass 2 plus country rules** (+0.0008).
3. **Country-agnostic model**, with state/legal code features hidden on 50% of training rows (+0.0006). This was the only work aimed
   at the unseen country that paid off.

Everything after that was +0.0001 or less on the leaderboard.

## What did not work

- **Tuning the decision thresholds on the training data.** It gained +0.0005 offline and **lost ~0.001** on the leaderboard. The extra
  links it added came disproportionately from the country missing in training: 14% of test records, 37% of the extra links.
- **Pretrained rerankers** (bge-reranker-v2-m3, two company-name models). All were worse than our small reranker fine-tuned on our own
  pairs (name-only top-1: 74% ours vs 68% / 65% / 39%).
- **Hiding the state code from the address reranker.** It barely used the state code (AUC 0.9910 → 0.9907), so there was nothing to gain.
- **Monotonic constraints, bigger embeddings, lower thresholds for name-only records**: flat or negative.

## Why we did not reach 0.985+

1. **Our validation did not look like the test set.** Test has a country that is absent from training (14% of records) and far more
   records without a true match (~40% vs 26%). Every offline number came from training countries only. The leave-one-country-out
   experiment (train on one country, score another) showed an unseen-country penalty of **0.025–0.03**. On 14% of test records that
   alone is roughly **0.004** of the ~0.0075 gap to the top. We found this late and never built the validation around it.
2. **Name-only records with identical names are a hard limit.** When a record has no address and several candidates share its exact
   name, every model, ours or pretrained, picks the right one only ~27% of the time. That is a data ceiling, not a model problem.
3. **Too many small bets, too late.** Most experiments in the last day were worth +0.0002 to +0.0005 offline, below what the
   leaderboard could even confirm. Each one took hours on a laptop that crashed when jobs overlapped.
4. **Offline wins were trusted over leaderboard evidence.** Offline results were reported as improvements before they were checked
   against the ways test differs from training. That cost at least one submission, and the cause of a drop was misdiagnosed twice.

## Do differently next time

- **On day one, compare train and test distributions:** countries, share of records with no match, records per group.
  Then build the offline validation to match test (leave-one-country-out, reweighting for records with no match), and tune every
  decision on that.
- **Treat a threshold or rule change as untrusted until it is checked on the validation that mimics test.** That applies most to changes
  that add links.
- **Spend submissions on large, different ideas early.** Use offline checks for the small ones.
- **Make generalization to unseen data the main workstream, not a late fix.** For example: text augmentation (transliteration,
  abbreviation and legal-form variants) so that a country's formats are learned from its patterns rather than from memorized code values.
- **Budget compute honestly.** Run one heavy job at a time on the laptop, and use the GPU box for anything over ~1 hour.
