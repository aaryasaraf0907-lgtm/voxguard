"""Drive Kaggle from a local terminal: upload code/data as private datasets, push GPU runs.

Requires the Kaggle CLI (``uv tool install kaggle --with six``) and a login
(``kaggle auth login``). Nothing here reads or prints credentials; the CLI handles them.

Typical flow::

    python -m voxguard.kaggle_runner upload-src          # code -> <owner>/voxguard-src
    python -m voxguard.kaggle_runner upload-prep         # manifests + codec variants
    python -m voxguard.kaggle_runner push --run rawnet2-clean --config configs/rawnet2.yaml \\
        --train-manifest asvspoof2019_la_train.csv --smoke
    kaggle kernels status <owner>/voxguard-rawnet2-clean
    kaggle kernels output <owner>/voxguard-rawnet2-clean -p runs/kaggle/rawnet2-clean

The private repo is shipped as a dataset (``git archive``) so kernels need no GitHub token.
Kernels locate their inputs by marker files, so they work whatever path Kaggle mounts them at.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

log = logging.getLogger(__name__)

STAGE = Path("data/processed/kaggle")  # gitignored
SRC_DATASET = "voxguard-src"
PREP_DATASET = "voxguard-prep"
LA_DATASET = "asvspoof2019-la"
SRC_PATHS = ["src", "configs", "pyproject.toml"]
SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{2,48}[a-z0-9]$")


@dataclass(frozen=True)
class RunSpec:
    run: str
    config: str  # repo-relative, e.g. configs/rawnet2.yaml
    train_manifest: str  # file name inside the prep dataset's manifests/
    dev_manifest: str = "asvspoof2019_la_dev.csv"
    epochs: int | None = None
    max_samples: int | None = None
    resume_run: str | None = None  # run whose last.pt to continue from


@dataclass(frozen=True)
class AugmentSpec:
    """CPU kernel: deterministic per-codec copies of a seeded subset of a prep manifest."""

    run: str
    manifest: str = "asvspoof2019_la_eval.csv"  # file name inside the prep dataset's manifests/
    subset: int | None = 10_000  # same seeded subset as datasets.load_rows(..., seed) locally
    seed: int = 0
    codecs: tuple[str, ...] = ("gsm", "amr_wb")  # codecs local ffmpeg lacks: held-out tests
    out_name: str = "eval_subset10k_heldout.csv"


def kaggle(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    cmd = ["kaggle", *args]
    log.info("$ %s", " ".join(cmd))
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


def kaggle_owner() -> str:
    out = kaggle("config", "view").stdout
    m = re.search(r"username:\s*(\S+)", out)
    if not m or m.group(1) in ("None", "YOUR_KAGGLE_USERNAME"):
        raise SystemExit("not logged in to Kaggle: run `kaggle auth login`")
    return m.group(1)


def kernel_slug(run: str) -> str:
    slug = f"voxguard-{run}"
    if not SLUG_RE.fullmatch(slug):
        raise ValueError(f"run name must give a slug of [a-z0-9-], 4-50 chars; got {slug!r}")
    return slug


def run_from_kernel(ref: str) -> str:
    """``owner/voxguard-<run>`` -> ``<run>``."""
    slug = ref.split("/")[-1]
    if not slug.startswith("voxguard-"):
        raise ValueError(f"not a voxguard kernel ref: {ref!r}")
    return slug.removeprefix("voxguard-")


def link_tree(src: Path, dst: Path) -> int:
    """Mirror ``src`` into ``dst`` with hard links (no extra disk). Returns files linked."""
    n = 0
    for path in src.rglob("*"):
        if path.is_file():
            target = dst / path.relative_to(src)
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                os.link(path, target)
            n += 1
    return n


def write_dataset_meta(folder: Path, owner: str, slug: str, title: str) -> None:
    meta = {"title": title, "id": f"{owner}/{slug}", "licenses": [{"name": "other"}]}
    (folder / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))


def publish_dataset(folder: Path, ref: str, message: str) -> None:
    """Create the dataset (private) or, if it exists, add a new version."""
    exists = kaggle("datasets", "status", ref, check=False).returncode == 0
    if exists:
        r = kaggle("datasets", "version", "-p", str(folder), "-m", message, "--dir-mode", "zip")
    else:  # `create` is private unless --public is passed
        r = kaggle("datasets", "create", "-p", str(folder), "--dir-mode", "zip")
    log.info(r.stdout.strip()[-500:])


def upload_src(owner: str, message: str) -> None:
    folder = STAGE / "src"
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    archive = subprocess.run(
        ["git", "archive", "--format=tar", "HEAD", *SRC_PATHS], capture_output=True, check=True
    ).stdout
    subprocess.run(["tar", "-x", "-C", str(folder)], input=archive, check=True)
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    (folder / "COMMIT").write_text(commit + "\n")
    write_dataset_meta(folder, owner, SRC_DATASET, "voxguard-src")
    publish_dataset(folder, f"{owner}/{SRC_DATASET}", message or f"code at {commit}")


def upload_prep(owner: str, message: str, processed: Path = Path("data/processed")) -> None:
    folder = STAGE / "prep"
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    n = link_tree(processed / "manifests", folder / "manifests")
    n += link_tree(processed / "codec_aug", folder / "codec_aug")
    log.info("staged %d files", n)
    write_dataset_meta(folder, owner, PREP_DATASET, "voxguard-prep")
    publish_dataset(folder, f"{owner}/{PREP_DATASET}", message or "manifests + codec variants")


KERNEL_TEMPLATE = '''\
"""Generated by voxguard.kaggle_runner. Do not edit by hand."""
import json, os, shutil, subprocess, sys
from pathlib import Path

import yaml

SPEC = json.loads({spec!r})
INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")


def find(marker):
    for depth in range(1, 5):
        for p in sorted(INPUT.glob("/".join(["*"] * depth))):
            if (p / marker).exists():
                return p
    sys.exit(f"no attached input contains {{marker}}")


src = find("src/voxguard/__init__.py")
la_root = find("LA/ASVspoof2019_LA_cm_protocols")
prep = find("manifests/" + SPEC["dev_manifest"])
print("code commit:", (src / "COMMIT").read_text().strip() if (src / "COMMIT").exists() else "?")
print("inputs:", src, la_root, prep, flush=True)

cfg = yaml.safe_load((src / SPEC["config"]).read_text())
out_dir = WORK / "runs" / SPEC["run"]
cfg["data"]["train_manifests"] = [str(prep / "manifests" / SPEC["train_manifest"])]
cfg["data"]["dev_manifests"] = [str(prep / "manifests" / SPEC["dev_manifest"])]
cfg["data"]["roots"] = [str(la_root), str(prep / "codec_aug")]
cfg["train"]["out_dir"] = str(out_dir)
cfg["train"]["num_workers"] = 4
cfg_path = Path("/tmp/run.yaml")
cfg_path.write_text(yaml.safe_dump(cfg))

cmd = [sys.executable, "-m", "voxguard.train", "--config", str(cfg_path), "--no-wandb"]
if SPEC["epochs"] is not None:
    cmd += ["--epochs", str(SPEC["epochs"])]
if SPEC["max_samples"] is not None:
    cmd += ["--max-samples", str(SPEC["max_samples"])]
if SPEC["resume_run"]:
    ckpt = "runs/" + SPEC["resume_run"] + "/last.pt"
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(find(ckpt) / ckpt, out_dir / "last.pt")
    cmd.append("--resume")

if shutil.which("nvidia-smi"):
    subprocess.run(["nvidia-smi", "-L"], check=False)
print("$", " ".join(cmd), flush=True)
env = {{**os.environ, "PYTHONPATH": str(src / "src")}}
sys.exit(subprocess.run(cmd, env=env).returncode)
'''


AUGMENT_TEMPLATE = '''\
"""Generated by voxguard.kaggle_runner (augment). Do not edit by hand."""
import json, os, subprocess, sys
from pathlib import Path

SPEC = json.loads({spec!r})
INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")


def find(marker):
    for depth in range(1, 5):
        for p in sorted(INPUT.glob("/".join(["*"] * depth))):
            if (p / marker).exists():
                return p
    sys.exit(f"no attached input contains {{marker}}")


src = find("src/voxguard/__init__.py")
la_root = find("LA/ASVspoof2019_LA_cm_protocols")
prep = find("manifests/" + SPEC["manifest"])
print("code commit:", (src / "COMMIT").read_text().strip() if (src / "COMMIT").exists() else "?")
env = {{**os.environ, "PYTHONPATH": str(src / "src")}}
py = [sys.executable, "-m"]
subprocess.run(py + ["voxguard.data.codecs"], env=env, check=True)

sys.path.insert(0, str(src / "src"))
from voxguard.data.datasets import load_rows  # noqa: E402
from voxguard.data.manifests import write_manifest  # noqa: E402

(WORK / "manifests").mkdir(parents=True, exist_ok=True)
subset = WORK / "manifests" / ("subset_" + SPEC["manifest"])
full = prep / "manifests" / SPEC["manifest"]
rows = load_rows([full], max_samples=SPEC["subset"], seed=SPEC["seed"])
print("subset rows:", write_manifest(rows, subset), flush=True)
out_manifest = WORK / "manifests" / SPEC["out_name"]
cmd = py + [
    "voxguard.data.augment", "--manifest", str(subset), "--data-root", str(la_root),
    "--out-root", str(WORK / "codec_eval"), "--out-manifest", str(out_manifest),
    "--config", str(src / "configs" / "codecs.yaml"),
    "--mode", "per-codec", "--no-clean", "--seed", str(SPEC["seed"]), "--workers", "4",
    "--codecs", *SPEC["codecs"],
]
print("$", " ".join(cmd), flush=True)
sys.exit(subprocess.run(cmd, env=env).returncode)
'''


def render_kernel(spec: RunSpec) -> str:
    return KERNEL_TEMPLATE.format(spec=json.dumps(asdict(spec)))


def render_augment_kernel(spec: AugmentSpec) -> str:
    return AUGMENT_TEMPLATE.format(spec=json.dumps(asdict(spec)))


def needs_internet(config: str | Path) -> bool:
    """True when the training config downloads pretrained weights (e.g. XLS-R from HF Hub)."""
    raw = yaml.safe_load(Path(config).read_text()) or {}
    return bool((raw.get("model") or {}).get("pretrained"))


def kernel_metadata(
    owner: str,
    spec: RunSpec | AugmentSpec,
    resume_from: str | None = None,
    gpu: bool = True,
    internet: bool = False,
) -> dict[str, object]:
    slug = kernel_slug(spec.run)
    return {
        "id": f"{owner}/{slug}",
        "title": slug,
        "code_file": "run.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": gpu,
        "enable_internet": internet,
        "dataset_sources": [
            f"{owner}/{SRC_DATASET}",
            f"{owner}/{PREP_DATASET}",
            f"{owner}/{LA_DATASET}",
        ],
        "kernel_sources": [resume_from] if resume_from else [],
        "competition_sources": [],
    }


def push(
    owner: str, spec: RunSpec, accelerator: str, resume_from: str | None, dry_run: bool
) -> Path:
    meta = kernel_metadata(owner, spec, resume_from, internet=needs_internet(spec.config))
    return _push_folder(spec.run, render_kernel(spec), meta, accelerator, dry_run)


def push_augment(owner: str, spec: AugmentSpec, dry_run: bool) -> Path:
    meta = kernel_metadata(owner, spec, gpu=False)
    return _push_folder(spec.run, render_augment_kernel(spec), meta, None, dry_run)


def _push_folder(
    run: str, code: str, meta: dict[str, object], accelerator: str | None, dry_run: bool
) -> Path:
    folder = STAGE / "kernels" / run
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    (folder / "run.py").write_text(code)
    (folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    if dry_run:
        log.info("dry run: kernel staged in %s", folder)
        return folder
    acc = ["--accelerator", accelerator] if accelerator else []
    r = kaggle("kernels", "push", "-p", str(folder), *acc)
    log.info(r.stdout.strip()[-500:])
    return folder


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(prog="python -m voxguard.kaggle_runner", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("upload-src", "upload-prep"):
        s = sub.add_parser(name)
        s.add_argument("-m", "--message", default="")
    s = sub.add_parser("push")
    s.add_argument("--run", required=True, help="run name, e.g. rawnet2-clean")
    s.add_argument("--config", required=True)
    s.add_argument("--train-manifest", required=True)
    s.add_argument("--dev-manifest", default="asvspoof2019_la_dev.csv")
    s.add_argument("--epochs", type=int, default=None)
    s.add_argument("--max-samples", type=int, default=None)
    s.add_argument("--smoke", action="store_true", help="shorthand: 1 epoch, 200 samples")
    s.add_argument("--resume-from", default=None, help="kernel ref whose output has last.pt")
    s.add_argument("--accelerator", default="NvidiaTeslaT4")
    s.add_argument("--dry-run", action="store_true", help="stage files, do not push")
    s = sub.add_parser("push-augment", help="CPU kernel: per-codec copies of a manifest subset")
    s.add_argument("--run", required=True, help="run name, e.g. eval-heldout")
    s.add_argument("--manifest", default="asvspoof2019_la_eval.csv")
    s.add_argument("--subset", type=int, default=10_000, help="seeded subset size (0 = all)")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--codecs", nargs="+", default=["gsm", "amr_wb"])
    s.add_argument("--out-name", default="eval_subset10k_heldout.csv")
    s.add_argument("--dry-run", action="store_true", help="stage files, do not push")
    args = p.parse_args(argv)

    owner = kaggle_owner()
    if args.cmd == "upload-src":
        upload_src(owner, args.message)
    elif args.cmd == "upload-prep":
        upload_prep(owner, args.message)
    elif args.cmd == "push-augment":
        aug = AugmentSpec(
            run=args.run,
            manifest=args.manifest,
            subset=args.subset or None,
            seed=args.seed,
            codecs=tuple(args.codecs),
            out_name=args.out_name,
        )
        push_augment(owner, aug, args.dry_run)
    else:
        spec = RunSpec(
            run=args.run,
            config=args.config,
            train_manifest=args.train_manifest,
            dev_manifest=args.dev_manifest,
            epochs=1 if args.smoke else args.epochs,
            max_samples=200 if args.smoke else args.max_samples,
            resume_run=run_from_kernel(args.resume_from) if args.resume_from else None,
        )
        push(owner, spec, args.accelerator, args.resume_from, args.dry_run)


if __name__ == "__main__":
    main()
