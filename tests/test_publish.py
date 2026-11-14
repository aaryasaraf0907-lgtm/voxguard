import json
from pathlib import Path

import pytest

from voxguard.publish import model_card

THR = {
    "model_version": "ssl_aasist/ssl-aasist-codec-aug-v2:best.pt:epoch7",
    "selected_on": "dev",
    "clean": 0.0123,
    "phone": 0.0456,
    "per_codec": {"opus": 0.05, "amr_nb": 0.07},
    "excluded_codecs": ["gsm"],
    "dev_eer": {"clean": 0.004, "phone": 0.02},
}
INFO = {"name": "ssl_aasist", "run": "ssl-aasist-codec-aug-v2", "epoch": 7}


def test_model_card_states_thresholds_privacy_and_attribution() -> None:
    hist = [{"epoch": 0, "train_loss": 0.9, "dev_eer": 0.37}]
    card = model_card(THR["model_version"], INFO, THR, hist)
    assert card.startswith("---\n") and "# VoxGuard: ssl-aasist-codec-aug-v2" in card
    assert "Confidential" in card and "do not make this repository public" in card
    assert "| EER threshold | 0.0123 | 0.0456 |" in card
    assert "| Dev EER | 0.40 % | 2.00 % |" in card
    assert "amr_nb 0.07, opus 0.05" in card and "held out): gsm" in card
    assert "ODC-By" in card and "| 0 | 0.9000 | 37.00 % |" in card


def test_publish_requires_thresholds(tmp_path: Path) -> None:
    from voxguard.publish import publish

    with pytest.raises(SystemExit, match="select-thresholds"):
        publish(tmp_path / "best.pt", "voxguard-team/x", dry_run=True)


def test_publish_dry_run_checks_the_api_accepts_the_file(tmp_path: Path, capsys) -> None:
    torch = pytest.importorskip("torch")
    from voxguard.models import build_model
    from voxguard.publish import publish

    params = {"gru_hidden": 16, "gru_layers": 1}
    torch.save(
        {"model": build_model("rawnet2", **params).state_dict(), "epoch": 2,
         "config": {"model": {"name": "rawnet2", **params},
                    "train": {"out_dir": "/kaggle/working/runs/rawnet2-clean-v2"}}},
        tmp_path / "best.pt",
    )  # fmt: skip
    thr = {**THR, "model_version": "rawnet2/rawnet2-clean-v2:best.pt:epoch2"}
    (tmp_path / "thresholds.json").write_text(json.dumps(thr))
    version = publish(tmp_path / "best.pt", "voxguard-team/voxguard-rawnet2-clean", dry_run=True)
    assert version == "rawnet2/rawnet2-clean-v2:voxguard-rawnet2-clean:epoch2"
    assert "# VoxGuard: rawnet2-clean-v2" in capsys.readouterr().out

    (tmp_path / "thresholds.json").write_text(json.dumps({**thr, "model_version": "x:y:epoch9"}))
    from voxguard.api.thresholds import ThresholdsError

    with pytest.raises(ThresholdsError):  # a file for another epoch would stop the API
        publish(tmp_path / "best.pt", "voxguard-team/voxguard-rawnet2-clean", dry_run=True)
