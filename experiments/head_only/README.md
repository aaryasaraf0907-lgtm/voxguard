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

## Round 2 (amendment, 22 Nov 2026 ~06:15 UTC, before any round-2 training)

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

## Round 2b (amendment, 22 Nov 2026 ~10:40 UTC, after the round-1 gate, before any round-2 result)

**What the round-1 gate showed** (16 candidates: 8 runs as A and B). All pass the two dev phone error
checks. All fail the 1.5x threshold check (round 2's target). **All also fail the In-the-Wild gate
EER check**: 15.33-18.87 % vs p3's 15.27 %; B (half p3) is consistently closer to p3 than A. With
Svarah, In-the-Wild false alarms at the candidate's own dev threshold pass (7.6-9.1 % vs 14.8 %);
without it they fail (16.0-23.7 %).

**The targeted fix.** In-the-Wild cannot be trained on: its 54 speakers are all assigned to the gate
and report parts. The one lever that needs no new data is how far each candidate leans on p3, whose
In-the-Wild EER the heads lose. So one candidate type is added: **C = 0.75 x p3's fake score + 0.25 x
the head's**, for every round-1 and round-2 head. The weight is fixed here and not tuned. No training
changes; round 2 runs as registered above.

**Judged the same way.** The same five gate checks, unchanged, on A, B and C of all 24 runs (72
candidates), the same choice rule among passing candidates, then the report, stop rules and test-only
checks. More candidates make a lucky pass more likely; the In-the-Wild report part and the eval
attacks, never used to choose, are the check on that.

## Round 2c (amendment, 22 Nov 2026 ~10:43 UTC, before any round-2 training)

**What failed.** The 8 candidates from the runs without Svarah fail the In-the-Wild false-alarm
check (16.0-23.7 % vs p3's 14.8 %); the 8 with Svarah pass it (7.6-9.1 %). In every pair,
Svarah also gave a lower dev phone EER and less threshold drift. The no-Svarah runs were there to
test whether Svarah helps; round 1 answered that.

**The fix.** The round-2 runs that start from a no-Svarah head train **with** Svarah (the same
genuine Svarah rows as the other runs; still no clones and no OpenVoice). Their names keep the base
run (`r2-<base>-mu<μ>`, so `r2-...-nosv-...` means "started from the no-Svarah head"). Nothing else
changes. The accent-shortcut check stays: the same-speaker clone test stop rule applies to any chosen
candidate, as above.

## Round 2d (amendment, 22 Nov 2026 ~10:55 UTC, before any round-2d training)

**Why.** Every round-1 candidate loses some In-the-Wild EER to p3, and a head on a frozen front-end
cannot learn new audio cues. The fallback CLAUDE.md section 9 already names, and `ssl_aasist.py`
already supports (`finetune: last_n`), is to train the top XLS-R layers too.

**What.** XLS-R's last **2** of 24 layers + its final layer norm + the head train together
(`--top`, `TopHead`), about 25M parameters, on the cached **inputs** of those 2 layers
(`features_top.py`: the same jobs and seeded waveforms as `features.py`, so row i is the same clip
and augmentation as in the head-only cache and `teacher.npy`). Checked before caching: p3's own top
layers + head on these inputs reproduce p3's scores to within fp16 rounding (largest difference
0.006 in P(fake) on 8 In-the-Wild clips).

**Settings (fixed now).** Start from p3 (its layers and head); with Svarah; λ 2 (distillation to p3);
lr 1e-5 for the XLS-R layers and layer norm, 5e-5 for the head; 3 epochs; the genuine anchor with
μ in {0.5, 2}; `--select-drift` (against p3's own threshold). 2 runs: `top-mu0.5`, `top-mu2`.

**Judged the same way.** One final gate and report on all 26 runs (8 round 1, 16 round 2, 2 round
2d), each as A, B and C (78 candidates), with the gate, choice rule, report, stop rules and test-only
checks unchanged.

**Execution order (22 Nov ~15:40 UTC, no change to any setting or rule).** The two round-2d runs
(`top-mu0.5`, `top-mu2`) run right after round-2 run 5 (`r2-l2-lr1e4-mu0.5`), before the remaining
11 round-2 runs, instead of after all of them, so the front-end question is answered sooner. Every
run, the gate, the choice rule and the report stay exactly as registered; all 26 runs are still judged
together at the end.

## Round 3 (amendment, 22 Nov 2026 ~15:50 UTC, before any round-3 training)

**Why.** Five round-2 head-only runs all stay 3.7-16x off p3's threshold, and every round-1 head loses
In-the-Wild EER: a head on a frozen front-end cannot learn new audio cues. Round 2d tries the top 2
layers on the laptop; round 3 is the full version, which needs a 16 GB GPU.

**What.** `scripts/exp_head_only/kaggle_full.py`: a **full fine-tune** of p3 (all of XLS-R + the
head) on a free Kaggle GPU (the team member's own account; private datasets: the experiment code,
p3's checkpoint, and the training audio, LA 2019 train + Svarah genuine; nothing else, no clones,
no OpenVoice, no dev, In-the-Wild, eval or Hindi audio).
- **Augmentation fresh every epoch** (seed = epoch and clip): a codec on the whole file with
  probability 0.6 (the head-only policy: G.711 µ/A, AMR-NB/WB, Opus weighted 2, noise before the
  codec on 30 %), a random 4 s crop, then `data/realistic.py` on both classes.
- **Loss:** inverse-frequency weighted CE + 2 x KL(p3 || model) at T=2 + 2 x the genuine anchor;
  p3 as a frozen fp16 teacher.
- **Optimiser:** Adam, lr 1e-6 for XLS-R (p3's own), 1e-5 for the back-end; batch 12; fp16 autocast;
  gradient checkpointing. Epochs: as many as fit a 10.5 h budget; every epoch saved.

**Choice and judgement (unchanged rules).** Each saved epoch is scored locally with the same code and
data as the gate (dev gate phone set, In-the-Wild gate part); the epoch is chosen by round 2's rule
(within 1.5x of p3's dev phone threshold first, then fewest missed fakes at 1 % false alarms, then
lowest EER), as A, B and C, and judged by the same five gate checks. Then the report, stop rules and
test-only checks, as above. It joins the same final comparison as rounds 1-2d.

**Round 3 additions (22 Nov ~16:05 UTC, still before any round-3 training).** Each one answers a
measured failure of the realistic fine-tune or a known risk of a full fine-tune:
- **RawBoost** (Tak et al., ICASSP 2022, the SSL-AASIST paper's augmentation), its LA setting
  (linear + non-linear convolutive noise, then impulsive signal-dependent noise), on half the clips
  after the codec and crop, before `data/realistic.py`. Our own implementation from the paper's
  description (`kaggle_full.py`, `rawboost`), peak level preserved.
- **The CNN feature encoder stays frozen** (p3's): only the 24 transformer layers, the layer norm and
  the back-end train, which keeps the low-level acoustic front end stable.
- **Candidate D = the average of the last 3 saved epochs' weights**, scored and judged like A, B and C
  (weight averaging usually generalises and calibrates better than any one epoch; it is also the
  check on unseen attacks, where the realistic fine-tune lost A10, A11, A17 and A18).

**Round 3 training safeguards (22 Nov ~16:25 UTC, still before any round-3 training).**
- **Layer-wise learning rates:** the top XLS-R layer at 1e-6 (p3's rate), each layer below at 0.9x
  the one above (bottom about 8e-8); the back-end at 1e-5. Lower layers carry general speech
  knowledge and move least, against the forgetting behind the realistic fine-tune's lost attacks.
- **Warm-up and cosine decay:** 500 steps of linear warm-up, then cosine to 10 % over 9 epochs.
- **Gradient clipping** at norm 1.0, and a batch with a non-finite loss is skipped and counted.
- **Candidate E = an exponential moving average of the weights** (decay 0.999), saved after every
  epoch (`ema_last.pt`, the final one is the candidate); judged like A-D.

**Round 3 on Colab (22 Nov ~17:10 UTC, before any round-3 training).** Kaggle's free T4 queue is
long tonight, so the **same** run (same script, settings and data, from the same private datasets)
is also started on Colab's free T4 as a backup, saving every epoch to Google Drive. The script now
takes its input and output folders from `INPUT_ROOT` / `OUT_DIR` and **resumes** after a disconnect
from the last saved epoch (weights and EMA; Adam restarts, the learning-rate schedule continues;
logged as a `resume` event). Rule for two results: the run that finishes its 10.5 h budget first
is round 3's result; if both finish, the other is reported as a **replication** (same settings,
different hardware and random order), never used to choose.
