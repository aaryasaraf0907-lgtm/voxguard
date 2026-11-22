# Head-only fine-tune of SSL-AASIST codec-aug p3 (experiment, off the submission path)

Pre-registered on 21 Nov 2026, **before any training**: the data, the candidates, the gates and
the stop rule are fixed here and in `splits/` (commit `04ae5c8`). The submission ships p3 whatever
this shows (Aarya, 21 Nov). Nothing from this branch is merged into `dev` before the submission;
Aarya decides afterwards whether to pick thresholds, publish or switch the demo.

## Why

p3 is the demo model. Measured limits it leaves (`docs/real-audio-testing.md`, sections 4–5):
genuine Opus calls end flagged (7 of 20 on the 2021 LA calls; p3's eval false-alarm rate is
highest on Opus, 0.84 % against 0.1–0.2 %), and same-speaker clones overlap a real voice.

The realistic fine-tune (`configs/ssl_aasist_realistic.yaml`) did not pass: its separation barely
changed (phone EER 1.94 % to 2.06 %) but the dev-chosen phone threshold more than doubled
(0.0127 to 0.0295) while eval genuine stayed low, so more fakes passed on every codec, and it lost
ground on eval attacks that dev does not contain (A11 46 to 91 % missed, A14 11 to 38 %, A10, A17,
A18 worse). This experiment is built around those two findings.

## What changes, and what does not

- **Frozen XLS-R.** Only the back-end (projection, encoder, AASIST graph: 348 K parameters) trains,
  from p3's weights. Features of p3's own front-end are cached once (fp16; the back-end on cached
  fp32 features reproduces p3's scores exactly, fp16 within about 0.5 %).
- **Data:** ASVspoof 2019 LA train only (A01–A06 and bona fide). Bona fide 4 copies, spoof 2 per
  clip; copy 0 without a codec, the others through a random codec (`codecs.yaml` bitrates, noise
  before the codec on 30 %), **Opus sampled twice as often** as G.711 µ-law, A-law, AMR-NB or
  AMR-WB. GSM stays the unseen codec. Then a seeded random 4 s crop and the realistic recording
  effects of `data/realistic.py` on **bona fide and spoof alike** (the realistic fine-tune's
  policy), so they carry no label information. No OpenVoice and no clones of anyone: the clone
  test stays a test on an unseen tool. **Svarah** (AI4Bharat, CC BY 4.0; amended 21 Nov, still
  before any training, once its terms were accepted): 4,880 clips of 2 s or more of genuine
  Indian-accented English, 2 copies each through the same codec and realistic pipeline, as
  bona fide only. Risk: with no Indian-accented fakes allowed, the head could learn "Indian
  accent = genuine", so every setting is trained **with and without Svarah**.
- **Loss:** class-weighted cross-entropy with p3's rule (weights = the inverse class frequency;
  p3's 0.9 / 0.1 is that for its data; here it is computed on the copies above) **plus distillation
  to p3** (KL divergence to p3's own back-end on the same features, temperature 2), so the head
  keeps p3's decisions where nothing pushes it away: the guard against losing eval attacks that
  dev cannot show.
- **Candidates:** A = the trained head (λ ∈ {0.5, 2}, lr ∈ {1e-4, 3e-4}, with and without
  Svarah, up to 8 epochs, best epoch on the dev gate below); B = the average of p3's and A's
  fake scores. 8 training runs. Choosing among them uses the dev gate and the In-the-Wild gate part only.

## Gate (all must hold, against p3 on the same cached features)

1. **2019 LA dev, phone** (µ-law, A-law, Opus pooled; `splits/la_dev_gate.txt`):
   EER no more than 0.10 percentage points above p3's, and missed fakes at a 1 % false-alarm rate
   no higher than p3's.
2. **Threshold drift:** the dev phone EER threshold (as `evaluate.py` picks it) within a factor of
   1.5 of p3's 0.0127, because a drifting threshold is what failed last time.
3. **In-the-Wild gate part** (27 speakers, `splits/itw_gate*.txt`): EER no higher than p3's, and
   false alarms at each model's own dev phone threshold no higher than p3's.

## Report (used once, at the end, never to choose)

2019 LA eval subset (`splits/la_eval_report.txt`; per codec and per attack, A10, A11, A17, A18
watched): EER, missed fakes at 1 % false alarms and at the dev threshold. In-the-Wild report part
(the other 27 speakers). Then, only for a candidate that passed the gate: the same-speaker clone
test, Set C and the ASVspoof 2021 LA calls (`live_eval.py`), all test-only. **Stop rule:** if a
watched attack's missed fakes at 1 % false alarms rise by more than 5 percentage points over p3,
the candidate is reported as not to adopt, whatever the gate said. The same holds if it catches
fewer of the same-speaker clones (as uploaded or over G.711) than p3: the check for the accent
shortcut above.

## Not used anywhere here

The Hindi set (untouched), the same-speaker clone test, Set C and the 2021 LA calls (test only),
In-the-Wild report speakers (report only).

## Round 2 (amendment, 22 Nov 2026 ~06:40 UTC, before any round-2 training)

**What round 1 showed so far.** Runs 1-3 of 8 have finished; runs 4-8, the gate, the In-the-Wild
gate part and the report have not been run. On the dev phone set, all three beat p3 on both error
checks (EER 0.18-0.26 % vs 0.47 %, missed fakes at 1 % false alarms 0.03 % vs 0.25 %). All three
fail one check only: the dev phone threshold moved 14x, 10.6x and 4.2x from p3's (limit 1.5x).
Stronger distillation (λ 2 vs 0.5) cut the drift, so the cause is the head's genuine scores leaving
p3's scale.

**The targeted fix.** One change aimed at that cause, everything else kept:
- **Warm start:** from each round-1 run's best head (`--init`), so what it learned is the
  starting point.
- **Genuine anchor:** a loss term on **training** rows only: μ x the mean squared difference
  between the head's and p3's logit margins (fake minus genuine) on the genuine rows of each
  batch (`--anchor`). Genuine scores keep p3's scale; fakes are free to move. No dev, In-the-Wild
  or eval data is used for training.
- **Settings:** the base run's λ and Svarah setting, lr 5e-5, 3 epochs, μ in {0.5, 2}: 2 round-2
  runs for each of the 8 round-1 runs (16 runs, named `r2-<base>-mu<μ>`). Fixed now, for all 8
  base runs, before runs 4-8 have finished.
- **Epoch choice** (`--select-drift`): among the epochs whose dev phone threshold is within 1.5x of
  p3's, the fewest missed fakes at 1 % false alarms, then the lowest EER; all epochs if none is.
  This uses the dev gate set as round 1's epoch choice does.

**Judged the same way, nothing relaxed.** The same gate, unchanged (all five checks, including the
1.5x threshold), on all 24 runs (8 round 1 + 16 round 2), each as A and B. Among the candidates that
pass, the one with the fewest dev phone missed fakes at 1 % false alarms, then the lowest dev phone
EER, is chosen. The report, the stop rules and the test-only checks follow for that candidate as
above. Running 16 more candidates makes a lucky gate pass more likely; the In-the-Wild report part
and the eval attacks, never used to choose, are the check on that.

**Fixes for other failures** (In-the-Wild, a watched attack, the clone test), if the gate or report
shows any, are not part of this round: they would need their own amendment, written before
training.

**Still apart from the submission.** The submission ships p3 (24 Nov); nothing here is merged before
then, and Aarya decides afterwards.
