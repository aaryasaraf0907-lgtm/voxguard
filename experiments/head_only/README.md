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

**Round 3 deviation (22 Nov ~17:10 UTC, seen in the Colab run's first log lines, before any epoch
finished).** Colab's ffmpeg has no AMR encoders (`libopencore_amrnb`, `libvo_amrwbenc`), so
`usable_codecs` skips AMR-NB and AMR-WB with a warning (as the repo intends for missing encoders):
training codecs are G.711 µ-law, G.711 A-law and Opus (weight 2), the same three the dev and eval
phone sets use. The Kaggle run logs its own codec list; the report states which were used.

## Adaptive round 2 (22 Nov ~17:40 UTC; the rule, before any adaptive run)

From here, each fix run targets the checks the previous runs failed, and is **written here before
it trains** (name `fix-<n>-…`, its settings and the failure it addresses). Fix runs go first; when
none is queued, the remaining pre-registered round-2 runs continue. No new run starts after 01:30
UTC on 23 Nov; the final gate and report then cover every completed run, with the gate, choice rule,
report and stop rules unchanged. Each adaptation uses only dev-gate results (per-epoch dev phone EER,
missed fakes, threshold) to decide, never the In-the-Wild report part, the eval attacks, the clone
test, Set C or the 2021 LA calls; those stay the honest check on how many adaptations were made.
(`train_head.py --top --init <head-only run>` now starts the back-end from that run's head under
p3's own top layers.)

**fix-1-top-init-l2 (22 Nov ~17:40 UTC, before it trains).** *Failures it addresses:* every head-only
run keeps its detection gain (dev phone missed fakes 0.03-0.09 %) but drifts 4-32x; the top-layer run
`top-mu0.5` holds the threshold near p3's (1.56-1.83x) but loses the gain (best epoch 0.26 % vs p3's
0.25 %). *Fix:* combine them: the back-end starts from round-1 head `l2-lr1e4` (the round-1 head with
the least drift, 4.2x, and 0.03 % missed fakes), XLS-R's top 2 layers start from p3 and train with it
(`--top --init l2-lr1e4`); λ 2, μ 2, lr 5e-5 (back-end) and 1e-5 (top layers), 3 epochs,
`--select-drift`. Judged at the final gate like every run.

**fix-2-top-init-l2-gentle (23 Nov ~00:00 UTC, before it trains; pushed in 52acbc8).** *Failure it addresses:* fix-1
beats p3 on dev phone EER (0.34-0.39 % vs 0.47 %) and missed fakes (0.12-0.22 % vs 0.25 %) in every
trained epoch, but its threshold jumps between epochs (2.01x, 1.65x, 3.26x of p3's), so the steps are
too large to settle inside 1.5x. *Fix:* the same start as fix-1 with gentler steps: lr 2e-5
(back-end) and 3e-6 (top layers), λ 2, μ 2, 3 epochs, `--select-drift`. Judged at the final gate.

**fix-r3 (round 3 again, 23 Nov ~00:00 UTC, before it trains; pushed in 52acbc8).** *Failure it addresses:* the first
round-3 run (Colab, 3 epochs before Colab ended the session) collapsed: dev phone EER 14.4 % after
epoch 1, recovering only slowly (7.2 %, 4.8 %), against p3's 0.47 %. *Cause, found without any test
data:* the run trained in PyTorch train mode. On 32 dev clips, p3's own fake-genuine logit margin in
train mode differs from its eval-mode margin by a mean squared 9.9 (6.9 from the AASIST back-end's
BatchNorm/dropout alone), and genuine clips often get "fake" margins. So the distillation and margin
anchors (to p3 in eval mode) chased that noise, and the BatchNorm running statistics drifted to the
augmented batches. *Fix:* train in eval mode (dropout and layerdrop off, BatchNorm statistics fixed
at p3's; gradients flow as before). Everything else in "Round 3" is unchanged (augmentation, loss,
learning rates, schedule, EMA, candidates A-E, the gate). A new output folder, so nothing resumes
from the first run. (The first run's scorer also printed its In-the-Wild preview numbers; they were
not used for this fix, which follows from the train/eval check above.) Its epochs are scored at the
final gate if they arrive in time, otherwise reported later in this PR, marked as late.

## Final gate (23 Nov, 02:40 UTC; all 23 completed runs, each as A, B and C: 69 candidates)

Produced by `gate_report.py <every completed run> --report` (private output
`runs/gate-report-final.json`). **Data sets:** "dev" is the **dev gate part** (`splits/la_dev_gate.txt`:
8,548 of the 24,844 LA dev clips, scored under G.711 µ-law, A-law and Opus, 25,644 scores), not the
full dev set, so p3's numbers here differ from `docs/results.md` (its EER threshold is 0.01485 here,
0.012676 on the full dev set). "ITW gate" is the In-the-Wild gate part (`splits/itw_gate.txt`: 27
speakers, 3,000 clips); its false alarms are counted at each candidate's own dev threshold.

| Candidate | Dev EER | Dev missed fakes at 1 % FA | Dev threshold (x p3) | ITW gate EER | ITW gate false alarms | Gate |
|---|---|---|---|---|---|---|
| p3 | 0.467 % | 0.25 % | 0.01485 (1x) | 15.27 % | 14.80 % | (reference) |
| **C:fix-2-top-init-l2-gentle** | **0.367 %** | **0.16 %** | 0.01932 (1.30x) | 14.60 % | 11.47 % | **pass** |
| C:top-mu0.5 | 0.478 % | 0.21 % | 0.01847 (1.24x) | 15.20 % | 12.40 % | pass |
| B:top-mu0.5 | 0.489 % | 0.24 % | 0.02136 (1.44x) | 15.07 % | 11.60 % | pass |
| A:fix-2-top-init-l2-gentle | 0.301 % | 0.14 % | 0.03150 (2.12x) | 14.53 % | 10.00 % | fail (threshold) |
| A:fix-1-top-init-l2 | 0.350 % | 0.12 % | 0.04838 (3.26x) | 15.53 % | 9.40 % | fail (threshold, ITW EER) |

The other 63 candidates fail at least one check. Three pass; the choice rule picks C:fix-2 (fewest
missed fakes).

**Report part for C:fix-2 (never used to choose).** Eval report part (`splits/la_eval_report.txt`:
7,200 LA eval clips under the same three codecs): EER 2.42 % (p3) vs 2.30 %, missed fakes at 1 % FA
10.1 % vs 7.6 %; watched attacks' missed fakes at 1 % FA: A10 64.0 vs 59.8, A11 29.9 vs 15.4, A17
0.9 vs 0.8, A18 12.7 vs 12.6 (so the attack stop rule does not trigger). In-the-Wild report part
(`splits/itw_report.txt`: the other 27 speakers, 3,000 clips): EER 15.47 % vs 14.87 %; at each
model's own dev threshold, false alarms 10.0 % vs 8.4 % and missed fakes 21.5 % vs 23.0 %.

## Test-only checks (23 Nov 02:48 UTC, commit dc9dcee: after the final gate, before any test-only result)

**Chosen candidate.** The final gate (all completed runs, unchanged checks) passed three
candidates: C:fix-2-top-init-l2-gentle, B:top-mu0.5 and C:top-mu0.5. The choice rule (fewest dev
phone missed fakes at 1 % false alarms, then lowest dev phone EER) picks
**C:fix-2-top-init-l2-gentle** (0.16 %, against 0.21 % and 0.24 %). The eval-attack stop rule does
not trigger (watched attacks' missed fakes at 1 % false alarms change by -4.3, -14.5, -0.2 and
-0.1 pp).

**How it is served.** `scripts/exp_head_only/blend_app.py`: the API's own detector loads p3, and
its model is wrapped so one XLS-R pass gives p3's P(fake) and the run's (layer-22 input through the
run's top layers and head), blended 0.75/0.25 exactly as in the gate. Checked against the gate's
cached-feature scores on 24 dev clips: largest difference 0.0037 (fp16 caching).

**Threshold.** C's dev phone EER threshold on the gate set is 0.01932, and p3's is 0.01485 there
against its deployed 0.012676 (chosen on the whole dev set by `evaluate.py`). The candidate is
served at the same ratio: 0.012676 x 0.01932 / 0.01485 = **0.01649** (1.30x). p3 runs at its
deployed 0.012676 (its `thresholds.json`). Nothing is tuned on the results below.

**Runs.** p3 and the candidate are both run now, the same way: a local scratch API (GPU, scratch
database, offline, tool code from origin/dev `89abfbf`), one after the other, on the same inputs.

1. **Same-speaker clone test** (`voice-clone-test/labels-clone.csv`: 20 genuine recordings of the
   owner's voice and 20 OpenVoice clones of it), `real_audio_sanity.py run`, as uploaded and over
   G.711 µ-law. **Stop rule (as registered above):** if the candidate flags fewer clones than p3,
   as uploaded or over G.711, it is reported as not to adopt.
2. **Set C** (12 consented genuine real-mic recordings), the same tool, as uploaded and over G.711
   µ-law: flagged genuine recordings, counts only.
3. **ASVspoof 2021 LA calls** (`live_eval.py`, the 19 Nov settings: 20 calls each of genuine, fake
   and mixed, about 20 s, speed 2, seed 0), codecs none, G.711 µ-law, G.711 A-law and Opus, plus
   **AMR-NB in place of GSM** (this machine's ffmpeg has no GSM encoder; AMR-NB, like GSM, was never
   in training, so it stays an unseen codec). Counts and medians only, as that tool reports them.

Reported as they come out, in this PR, whatever they show.

## Test-only results (23 Nov, 02:49-04:00 UTC): C:fix-2 is **not to adopt**

Both models on the same local scratch API (GPU, tool `89abfbf`), p3 at 0.012676, C:fix-2 at 0.01649.

| Check | p3 | C:fix-2 |
|---|---|---|
| Clone test, as uploaded: clones caught / genuine flagged (of 20 each) | 15 / 1 | **12** / 0 |
| Clone test, G.711 µ-law | 16 / 2 | **12** / 0 |
| Set C (12 genuine real-mic recordings), flagged as uploaded / G.711 | 0 / 0 | 0 / 0 |
| 2021 LA live calls, genuine calls with a live alert (5 codecs x 20) | 2 of 100 | 3 of 100 |
| 2021 LA live calls, fake calls alerted | 100 of 100 | 100 of 100 |
| 2021 LA live calls, mixed calls alerted / alerts before the switch | 100 / 8 | 100 / 2 |
| 2021 LA live calls, flagged verdicts on genuine calls | 16 of 100 | 13 of 100 |

**Stop rule triggered:** C:fix-2 catches fewer of the same-speaker clones than p3 (12 vs 15 as uploaded,
12 vs 16 over G.711), so it is reported as not to adopt, as registered.

*Why, for the record (not used to change the verdict):* at its higher threshold it flags no genuine
recording of this voice, and at p3's threshold it catches all 40 clones but flags 26 of 40 genuine
ones. Ranked by score, 14 clones (as uploaded) and 13 (G.711) sit above every genuine recording,
against p3's 15 and 13: on this voice it separates clones from genuine no better than p3; it trades
missed clones for false alarms along the same line. On the 2021 LA calls it is about level with p3.
The dev and eval gains (missed fakes 0.25 % to 0.16 % on dev, A11 29.9 % to 15.4 %) do not carry over
to the clone test, the same-speaker check this rule was written for. The submission ships p3, as
planned. Round 3 (full fine-tune, `fix-r3` on Kaggle): see "Round 3 results" below.

## Round 3 results (23 Nov, 10:46-11:05 UTC; reported late, after the final gate): **not to adopt**

Both Kaggle runs used the same script, data and budget (10.5 h on a T4, as many epochs as fit);
the only code difference is `fix-r3` (eval mode) and a GPU-memory log line.

| Run | Epochs | Training loss, epoch 1 → last | Dev phone EER | In-the-Wild gate EER |
|---|---|---|---|---|
| Pre-fix (train mode, `voxguard-exp-full`) | 13 | 16.56 → 5.34 | 8.03 % (epoch 13), 7.72 % (E) | 51.0 %, 50.0 % |
| `fix-r3` (eval mode, `voxguard-exp-full-r3b`) | 14 | 0.73 → 0.67 | 0.454 % (epoch 14, E) | 15.51 % |

Same loss formula in both runs. The pre-fix run collapsed as round 3 v1 did, so the train-mode
diagnosis holds: with only that change, the run stays at p3's level.

**Gate, `fix-r3`.** Epochs 4, 9 and 14 and E (`ema_last`), each as A, B and C: 12 candidates. p3 is
scored on the same rows (its cached features, as in every gate above); the r3b checkpoints are scored
by `score_full.py` (whole model, fp16).

| Candidate | Dev EER | Dev missed fakes at 1 % FA | Dev threshold (x p3) | ITW gate EER | ITW gate false alarms | Gate |
|---|---|---|---|---|---|---|
| p3 (same rows) | 0.585 % | 0.36 % | 1x | 15.85 % | 12.34 % | (reference) |
| A: epochs 4 / 9 / 14 / E | 0.454-0.502 % | 0.17-0.22 % | 3.28-4.21x | 14.87-15.51 % | 7.91-8.86 % | fail (threshold) |
| B: epochs 4 / 9 / 14 / E | 0.454-0.519 % | 0.20-0.22 % | 2.17-2.54x | 15.51-16.14 % | 8.86-9.49 % | fail (threshold; B:14 also ITW EER) |
| C: epochs 4 / 9 / 14 / E | 0.474-0.558 % | 0.22-0.28 % | 1.65-1.79x | 15.85 % | 9.49-9.81 % | fail (threshold) |

No candidate passes, so the test-only checks do not run. The fine-tune ranks the dev phone clips
better than p3 on these rows (EER 0.454 % vs 0.585 %, missed fakes 0.17 % vs 0.36 %, on 5,129 scores);
on In-the-Wild it is no worse than p3: on 600 clips the In-the-Wild differences, gains and losses
alike, are within noise (about ±1.5 pp). But its scores move to a different scale: the dev threshold
moves from 0.0173 (p3, these rows) to 0.057-0.073, 3.3-4.2x or 1.2-1.5 in log-odds, and even the
0.75/0.25 blend stays at 1.65x or more, the same failure as round 1.

**Deviations from the registered plan, stated plainly.** (1) Measured on every 5th row of both gate
sets (5,129 dev phone scores, 600 In-the-Wild clips), not all rows. A full-set scoring of the two
closest checkpoints was started and stopped by the team: the verdict cannot change what ships, and
thresholds 1.65x and above are not near the 1.5x line. (2) Epochs 4, 9 and 14 and E only, not every
epoch, and D (the last-3 average) not built. The scored epochs barely differ (dev EER 0.454 % at epochs
9, 14 and E), so an unscored epoch is unlikely to sit inside 1.5x; this is a judgement, not a
measurement. Scores and the scoring script are kept privately with the other runs.

**Outcome of the experiment.** No candidate from rounds 1-3 is adopted. The submission ships p3,
unchanged.
