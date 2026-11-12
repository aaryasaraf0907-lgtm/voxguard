"""Streamlit settings for the demo stay the same for `streamlit run` and the Docker images."""

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_local_config_hides_developer_toolbar() -> None:
    config = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text())
    assert config["client"]["toolbarMode"] == "viewer"  # no Deploy / Rerun / Clear cache


def test_docker_images_hide_developer_toolbar() -> None:
    # The images do not copy .streamlit/config.toml; they set Streamlit options by env var.
    for dockerfile in ("Dockerfile.ui", "Dockerfile.space"):
        text = (ROOT / "docker" / dockerfile).read_text()
        assert "STREAMLIT_CLIENT_TOOLBAR_MODE=viewer" in text, dockerfile
