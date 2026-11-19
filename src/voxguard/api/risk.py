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
