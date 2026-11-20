# Demo script

How to present LineTrue live (about 8 minutes) and record the demo video (about 3 minutes). It
uses the local Docker Compose deployment with the real detector
([M2 demo runbook](deployment.md#m2-demo-runbook-real-model-in-compose)).

**Numbers are not repeated here.** For results, show [`docs/results.md`](results.md), which
scripts generate (CLAUDE.md section 10). On screen, show only what the app itself displays.

## Clips to prepare

No audio is committed; keep the clips in a private folder outside the repo. **Score each clip
beforehand** with the [sanity tool](real-audio-testing.md) against the same model and threshold,
and pick only clips that behave as the script needs.

| Role | What | Requirement |
|---|---|---|
| **G1, genuine** | A consented recording of the presenter, 10–30 s, quiet room, saved as M4A by the laptop recorder | Not flagged, clearly below the threshold, both as uploaded and through G.711 |
| **G2, genuine (backup)** | An ASVspoof 2019 or 2021 LA eval bona fide file | As G1, with a wider margin |
| **F1, F2, fake** | ASVspoof 2019 LA eval spoofs (the model's training domain; ODC-By, cite Todisco et al., Interspeech 2019) | Flagged, far above the threshold (no caution), both as uploaded and through G.711 |
| **N1, near the threshold** | A consented genuine recording that the model flags with the caution | Flagged, below 1.5× the threshold, so the yellow caution shows |
| **M, mixed call (timeline and live)** | Constructed privately: G1 (or another consented genuine recording) followed by F1, joined into one 16 kHz file | Timeline flags it only after the fake part starts, and the genuine part stays below the threshold. **Also check it live** (`python scripts/stream_call.py M.wav` against the demo stack): no alert and no "elevated" during the genuine part, the alert a few seconds after the switch. A transfer checked during the genuine part must come back **allow** |

**Getting candidates.** Any ASVspoof 2019 LA eval file can be fetched on its own from the official
`LA.zip` (it supports HTTP range requests), so a handful of clips is a few MB instead of the 7.6 GB
archive. Labels come from the official eval key only, never from scores. Fakes from ASVspoof 2021
LA work less reliably: its transmission conditions are outside training, and several of its spoofs
score near or below the threshold. Pick from 2019 for F1 and F2.

**Choosing.** Score every candidate as uploaded and with `--codec g711_mulaw`, then keep only clips
whose decision is the same in both conditions and far from the threshold (for example, fakes at
several times the threshold, genuine clips clearly below it). Scores are deterministic for the same
file and model, so a clip chosen this way behaves the same on the day. Choose again whenever the
model, the threshold or the scoring changes (for example #103, which now scores 4–5 s clips on
two windows: re-check N1 and any clip of that length).

M is constructed: say so when showing it ("a genuine recording followed by an AI-generated
segment, joined for the demo"). It shows when the running average crosses the threshold, not a
measured detection delay.

N1 is a genuine voice wrongly flagged. Present it as exactly that: the honest limit of the model
on off-domain speech, and the reason for the caution.

**No same-speaker clone pair** (a presenter's recording next to an AI clone of the same voice)
unless the final model catches the clone: on the current model it did not separate them reliably
(`docs/real-audio-testing.md`, section 5). Re-check after the final model.

## Before the demo (T−30 min)

1. Start the stack and run the pre-flight checks (runbook steps 1–2): `/health` shows the real
   model, the log shows the phone threshold, `/v1/codecs` lists G.711, GSM and Opus.
2. Reset the demo data (runbook step 4), so the lists start empty.
3. **Warm up:** analyze G1 (not flagged, and with no caller ID, so the flagged sessions, blocklist
   and alerts stay empty), then analyze it once more, within the last minute or two before
   presenting, and again before the live part if the stack has been idle for ten minutes
   or more. Keep the laptop plugged in, on a performance power mode, with heavy apps closed.
   The first request after an idle period can be slower (see the runbook's idle-latency
   measurement); the next one is back to normal.
4. Open the UI (http://localhost:8501) in the browser you will present from. Check the sidebar
   says **Connected** and shows the model. Set the zoom so the result metrics are readable on the
   projector.
5. Have the clips in one folder, a terminal with the runbook open, and the backup video ready.

## Live run (about 8 minutes)

| # | Time | Do | Say |
|---|---|---|---|
| 1 | 0:00 | Title screen or the README diagram | Scam calls use cloned voices. Real calls pass through phone codecs that remove the artifacts most detectors are trained on, so a detector that is accurate in the lab can fail on a phone line. |
| 2 | 0:45 | **Analyze audio** → **Upload a file** → G1 → **Analyze** | A genuine recording: **Not flagged**. Point at the fake score, the threshold and the detector latency. |
| 3 | 1:45 | F1 → **Analyze** | An AI-generated voice: **Flagged**, red banner. The score is far above the threshold. |
| 4 | 2:30 | Turn on **Simulate phone line**, codec **G.711 µ-law** → **Analyze** F1 again | The same clip after a real telephone codec. Both results appear side by side; the decision holds on the phone line. This is the point of the codec-aware training. |
| 5 | 3:15 | Turn on **Show live-monitoring timeline**, phone line off → M → **Analyze**; press **Replay** | What a live call monitor would have shown: each 4 s window and the running average against the threshold. The genuine part stays below it; the alert comes a few seconds after the fake voice starts. Same verdict as without the timeline. |
| 6 | 3:40 | Sidebar → **Open the live call monitor** → choose M → **Play and monitor** (or **Start listening** and play M from a phone next to the laptop) | The same mixed call, live: the score updates every 2 s while it plays. The genuine part stays below the threshold; a few seconds after the fake voice starts, the banner turns red: *do not approve payments, hang up and call back on a number you trust*, with what to do next (ask something only the real person knows; report at 1930 or cybercrime.gov.in). Point at the consent notice: the other party is told the call is checked. A single window is never enough to warn: it waits for two, and for the call's risk to reach 1.5 x the threshold; just over the threshold the page only says the risk is elevated (genuine calls often sit there). Its stored score is the average of overlapping live windows, so it will not equal the score of the same clip uploaded in step 5: do not compare the two on stage. Backup: `python scripts/stream_call.py M.wav` in a terminal. |
| 7 | 4:20 | Still on the monitor, while M plays: **Approve transfer** (₹75,000) once during the genuine part, again after the warning; record the two steps under **two different names** (e.g. the presenter for the call-back, a teammate as second approver); open the incident report and scroll to **Evidence trail** | During the genuine part the transfer is allowed. A few seconds into the fake voice the same transfer is **held**: it needs a call-back on a known number and a second approver, and the two must be different people (the same name twice is refused), the standard bank defence against voice fraud. Each decision and step is logged; the report keeps scores and decisions, never audio, and its evidence trail says **Intact**: every record is hash-chained, so an edit afterwards would show. |
| 8 | 5:00 | Show `docs/results.md`, the 2×2 matrix | The measured result: trained on clean audio vs codec-augmented, tested clean vs phone-degraded. Read the two phone-degraded cells. |
| 9 | 5:30 | Enter a **Caller ID** (e.g. `+91-DEMO-0001`) → F2 → **Analyze**; open **Flagged sessions**, **Blocklist**, **Alerts** | Prevention: a flagged call is stored, an alert is raised, and the caller is added to the blocklist with a hit counter. Analyze F2 once more with the same ID to show the counter go up. |
| 10 | 6:20 | N1 → **Analyze** | A genuine voice, flagged, but with the yellow caution "Close to the decision threshold". Flags just above the threshold are hints, not verdicts. Off-domain speech (unusual microphones, rooms, languages) is the model's main weakness; we report it rather than hide it. |
| 11 | 7:05 | Sidebar → **हिन्दी** | The interface in Hindi. Detection was trained and evaluated in English; Hindi detection is only an evaluated generalisation test, not a supported feature. |
| 12 | 7:40 | Back to English; summarise | The detector, the phone-line evaluation, the prevention layer, the latency on CPU. Questions. |

Uploaded audio is never stored, and the live call monitor keeps only the latest 4 s in memory:
only each session's score, threshold, label, flag, codec, model
version, latency and optional caller ID.

## If something goes wrong

| Problem | Do |
|---|---|
| Sidebar shows **API unavailable** | `docker compose ps`; if the API is not running, `docker compose up -d --wait` (runbook step 5). Fill the wait by continuing with the slides. |
| A request is slow | Wait: the first request after an idle period can take a few seconds, and the next one is back to normal. Say that the latency shown is the detector's time on a CPU laptop, and analyze the clip once more if it looks out of line. |
| A result is not what the script expects | Say so: the model makes mistakes, and N1 already shows one. Move on with the next clip. |
| Nothing works | Play the backup video. |

The demo runs offline once the model is in the volume; no network is needed.

## Likely questions

- **What is new?** The codec-aware training and evaluation regime, not the architecture
  (XLS-R + AASIST is published work): random telephone codecs during training, and evaluation on
  clean and phone-degraded audio, including a codec never used in training (GSM).
- **Why is the threshold so low?** It is chosen on the development set at the equal error rate,
  separately for clean and phone audio. The model's scores for real speech sit in a narrow low
  band, so the operating point is low; only the comparison with the threshold matters.
- **How often does it flag real people?** Point to the out-of-domain evaluation in
  `docs/results.md`, the live and same-speaker counts in `docs/real-audio-testing.md`
  (sections 4 and 5), and to N1. They are counts on small sets: do not quote them as rates.
- **Does it work in Hindi?** Claim only what `docs/results.md` reports for the Hindi test set.
- **Is it real time?** Show the detector latency in the result and the latency table in
  `docs/results.md`: each 4 s window is scored faster than it plays, on a laptop CPU. The
  timeline shows when a live monitor would have flagged the call; the demo scores a recording,
  it is not connected to a phone line. The live call monitor (step 6) does score while the
  audio plays, from a microphone or a file. Capacity under load is in `docs/deployment.md`,
  "Load test".
- **How would it reach real calls?** Only through whoever owns the line. A bank's or
  company's own phone system (PBX, contact-centre platform) can send a copy of a call's audio to
  `/v1/stream` in real time, with the usual "this call may be monitored" notice. A telecom
  operator could do it in the network under a legal framework, and approved bots can do it on
  meeting platforms. The demo listens through a microphone or plays a file; it does not tap
  anyone's calls.
- **Can the records be changed afterwards?** Not without it showing. Every stored session, a
  live call's window scores, each transfer decision and each verification step are chained by
  SHA-256; `/v1/evidence/verify` recomputes the chain from the records, and the chain head goes
  out with every webhook, so a copy is held elsewhere. Tamper-evident, not tamper-proof, and no
  claim of legal admissibility.
- **Would it catch a clone of my own voice?** Not reliably yet. We tested it on a team member's
  own voice and an open cloning tool (`docs/real-audio-testing.md`, section 5): most clones scored
  above the genuine recordings, but some overlapped them, live especially. Say that this is the
  hardest case and what the final model was trained to improve; quote only the counts in that
  section.
- **Why did it say "elevated" and not alert?** Genuine speech often runs just over the
  threshold, so the live alert waits until the call's risk reaches 1.5 x it
  (`docs/real-audio-testing.md`, section 4). The stored verdict still uses the threshold.
- **What would deployment need?** That telephony integration, logins for the operators who
  record verification steps, and the alert webhook connected to the bank's own SMS, e-mail or
  security systems. Real OTP sending and core-banking holds are outside this prototype.

## Video (about 3 minutes)

Record the screen at 1080p with the presenter's voice; no other people's voices or data.

| Shot | Length | Content |
|---|---|---|
| 1 | 20 s | The problem, over the architecture diagram |
| 2 | 30 s | G1 not flagged, then F1 flagged |
| 3 | 30 s | F1 on the simulated phone line, side by side |
| 3b | 20 s | M with the timeline and Replay: the alert after the fake part starts |
| 3c | 25 s | M on the live call monitor: below the threshold, then the red warning while it plays |
| 3d | 25 s | The transfer approved during the genuine part, then held after the warning; two different people release it; the report's evidence trail says Intact |
| 4 | 30 s | The 2×2 matrix in `docs/results.md` |
| 5 | 35 s | Prevention: caller ID, flagged sessions, blocklist, alerts |
| 6 | 15 s | N1 with the caution: the honest limitation |
| 7 | 15 s | Hindi interface |
| 8 | 20 s | Summary and limitations |

Keep the repository and video private until the IPR cell confirms the provisional filing
(CLAUDE.md section 15).
