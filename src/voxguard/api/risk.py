"""Risk constants shared by the live monitor (api/stream.py) and the transfer policy
(api/policy.py)."""

# No alert on a single window: on the real model, the first window of a genuine recording sat
# just over the threshold (0.012836 against 0.012676) and the next ones well below it, so the
# alert waits for the running average of two windows (about 6 s of audio). Measured on a
# genuine-then-fake call: alert 1.3 s after the fake part began, none during the genuine part.
MIN_ALERT_WINDOWS = 2
# Graded warning, the UI's near-threshold caution band (agreed 16 Nov, ui/app.py): just over the
# threshold the page asks to verify the caller another way; well over it, to hang up.
CAUTION_FACTOR = 1.5
# The live alert (the alert message, the live_alert webhook, the call's first_flag_s) fires when
# the running average reaches ALERT_FACTOR x the threshold, not the threshold itself. Measured on
# 300 streamed ASVspoof 2021 LA calls and a same-speaker voice-clone test (20 Nov, #101): with the
# alert at the threshold, 35 of 100 genuine calls raised one (the running average of genuine
# speech often sits just over it); at 1.5 x, 2 of 100, and all 100 fake calls still alerted at
# 6.25 s. Between the threshold and this level the page shows that the risk is elevated, without
# an alert. The stored verdict still uses the threshold (the prevention rule is unchanged).
ALERT_FACTOR = CAUTION_FACTOR
