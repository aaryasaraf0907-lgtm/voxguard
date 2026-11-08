# Training on Kaggle (or Colab) free GPUs

Exact steps to run VoxGuard data prep and training from this **private** repo. Facts marked
*(verified)* were checked in a real Kaggle notebook on 8 Nov 2026.

## 0. One-time setup

1. Kaggle: verify your phone number (Settings), or Internet and GPUs stay disabled.
2. GitHub: Settings → Developer settings → Personal access tokens → **Fine-grained token**.
   Repository access: only `voxguard`; permission **Contents: Read-only**; expiry ~30 days.
   No GitHub Developer Program registration is needed.
3. Never paste the token into a cell, a file, or the repo. Delete it when the project ends.

## 1. Get the dataset onto Kaggle

Edinburgh DataShare returns **HTTP 403 to Kaggle and Colab IP ranges** *(verified: even the
landing page is 403)*, so the script cannot download it there. Instead:

1. On your own machine: `uv run python -m voxguard.data.download --yes --keep-zip`
   (7.64 GB; resumable. DataShare caps per-client throughput at roughly 0.3–0.6 MB/s
   *(verified)*, so expect several hours; parallel connections do not help).
2. kaggle.com/datasets → **New Dataset** → upload `data/raw/LA.zip` → visibility **Private**.
   Kaggle may unpack the zip on upload; both cases are handled below.
3. In the notebook: **Add Input** → your dataset. It appears under `/kaggle/input/<name>/`.

Licence: ODC-By (attribution; redistribution under the same terms). Keep the copy private
anyway (CLAUDE.md section 15). Cite Todisco et al., Interspeech 2019.

## 2. Notebook settings

- Session options → **Internet: On**.
- Accelerator: **None** for data prep; **GPU T4 x2** or **P100** for training.
- Add-ons → **Secrets** → add `GITHUB_TOKEN` (and `WANDB_API_KEY` if you use W&B), and tick
  each one so it is attached to this notebook.

## 3. Clone (cell 1)

```python
from kaggle_secrets import UserSecretsClient
import subprocess

tok = UserSecretsClient().get_secret("GITHUB_TOKEN")
r = subprocess.run(
    ["git", "clone", "-b", "dev", "--depth", "1",
     f"https://x-access-token:{tok}@github.com/aaryasaraf0907-lgtm/voxguard.git",
     "/kaggle/working/voxguard"],
    capture_output=True, text=True)
print("clone exit code:", r.returncode)
print(r.stderr.replace(tok, "***")[-300:])  # never print the raw token
```

## 4. Environment (cell 2)

Kaggle ships Python 3.13, ffmpeg 6.1.1, numpy, pyyaml, soundfile, torch and transformers
*(verified, except torch/transformers versions: check below)*. Run modules with
`PYTHONPATH=src` instead of `pip install`, which would try to reinstall torch.

```bash
%cd /kaggle/working/voxguard
!python --version && git log --oneline -1
!python -c "import torch, transformers, soundfile, yaml; print(torch.__version__, transformers.__version__, torch.cuda.is_available())"
!df -h /kaggle/working /tmp
!PYTHONPATH=src python -m voxguard.data.codecs
```

Disk: `/kaggle/working` is 20 GB and is what a saved version keeps; `/tmp` has hundreds of
GB but is wiped when the session ends *(verified)*. Put raw audio in `/tmp`.

## 5. Manifests (cell 3)

If the dataset is still a zip:

```bash
!PYTHONPATH=src python -m voxguard.data.download --from-zip /kaggle/input/<name>/LA.zip \
    --root /tmp/raw --manifests-dir /kaggle/working/manifests
```

If Kaggle unpacked it (you see an `LA/` folder):

```bash
!PYTHONPATH=src python -m voxguard.data.download --manifests-only \
    --root /kaggle/input/<name> --manifests-dir /kaggle/working/manifests
```

Check: `!wc -l /kaggle/working/manifests/*.csv` (about 25.4k train, 24.8k dev, 71.2k eval
rows plus a header each).

## 6. Codec-degraded variants (cell 4)

Test on 200 files first and check the size before the full run:

```bash
!PYTHONPATH=src python -m voxguard.data.augment \
    --manifest /kaggle/working/manifests/asvspoof2019_la_train.csv \
    --data-root /tmp/raw --out-root /kaggle/working/processed \
    --out-manifest /kaggle/working/manifests/train_codec_aug.csv \
    --mode variants --n-variants 2 --seed 0 --max-files 200 --workers 4
!du -sh /kaggle/working/processed
```

Use the root that holds the `LA/` folder as `--data-root` (`/tmp/raw` or `/kaggle/input/<name>`).
Eval copies (`--mode per-codec`) for all 71k files are large: use a subset first.

Then **Save Version → Save & Run All**, and turn the outputs (`manifests/`, `processed/`)
into a private Kaggle Dataset so training notebooks can attach them.

## 7. Training (GPU notebook)

Point the config at Kaggle paths with a small override file, e.g.
`/kaggle/working/kaggle_rawnet2.yaml`:

```yaml
model: {name: rawnet2}
data:
  train_manifests: [/kaggle/input/<prep-dataset>/manifests/asvspoof2019_la_train.csv]
  dev_manifests: [/kaggle/input/<prep-dataset>/manifests/asvspoof2019_la_dev.csv]
  roots: [/kaggle/input/<la-dataset>, /kaggle/input/<prep-dataset>/processed]
train: {epochs: 20, batch_size: 32, num_workers: 2, out_dir: /kaggle/working/runs/rawnet2_clean}
```

```bash
!PYTHONPATH=src python -m voxguard.train --config /kaggle/working/kaggle_rawnet2.yaml --no-wandb
```

Smoke test first: add `--max-samples 200 --epochs 1`.

SSL-AASIST: copy the model section from `configs/ssl_aasist.yaml`. XLS-R (1.27 GB) downloads
from Hugging Face on first use. If you run out of GPU memory, switch to
`finetune: last_n`, `last_n: 4`, `frontend_lr: 1.0e-5`, `lr: 1.0e-4`.

Checkpoint size: RawNet2 ~100 MB; SSL-AASIST ~3.6 GB per file with full fine-tuning
(weights + Adam state), so `last.pt` + `best.pt` is ~7 GB of the 20 GB working disk.

## 8. Resuming after a session dies

Sessions are time-limited and can stop. `last.pt` is written atomically every epoch.

1. Save a version (or it is lost), then add that notebook's output as an input of the next run.
2. Copy the checkpoint back and resume:

```bash
!mkdir -p /kaggle/working/runs/rawnet2_clean
!cp /kaggle/input/<previous-run>/runs/rawnet2_clean/last.pt /kaggle/working/runs/rawnet2_clean/
!PYTHONPATH=src python -m voxguard.train --config /kaggle/working/kaggle_rawnet2.yaml --resume --no-wandb
```

## 9. W&B (optional)

```python
import os
os.environ["WANDB_API_KEY"] = UserSecretsClient().get_secret("WANDB_API_KEY")
```

Set `wandb: {enabled: true}` in the config and drop `--no-wandb`.

## Colab differences

- Secrets: key icon in the left sidebar; read with `from google.colab import userdata;
  userdata.get("GITHUB_TOKEN")`.
- Storage: mount Google Drive (`from google.colab import drive; drive.mount("/content/drive")`)
  and put `out_dir` on Drive so checkpoints survive disconnects.
- DataShare is blocked from Colab too *(verified)*: upload `LA.zip` to Drive instead.
