"""configure_logging: VoxGuard's startup INFO lines are visible when served by uvicorn."""

import logging

from voxguard.api import main

PACKAGE = logging.getLogger("voxguard")


def _bare_logging(monkeypatch) -> None:
    """Logging as uvicorn leaves it for our package: no root handler, nothing on voxguard.
    Called inside the test body: pytest adds its capture handler to root per test phase."""
    monkeypatch.setattr(logging.getLogger(), "handlers", [])
    monkeypatch.setattr(PACKAGE, "handlers", [])
    monkeypatch.setattr(PACKAGE, "level", logging.NOTSET)


def test_adds_one_info_handler_to_the_package_logger(monkeypatch) -> None:
    _bare_logging(monkeypatch)
    main.configure_logging()
    main.configure_logging()  # idempotent
    assert len(PACKAGE.handlers) == 1 and PACKAGE.level == logging.INFO
    assert logging.getLogger("voxguard.api.detector_loader").isEnabledFor(logging.INFO)
    assert PACKAGE.handlers[0].formatter._fmt == main.LOG_FORMAT


def test_leaves_already_configured_logging_alone(monkeypatch) -> None:
    monkeypatch.setattr(logging.getLogger(), "handlers", [logging.NullHandler()])
    monkeypatch.setattr(PACKAGE, "handlers", [])
    main.configure_logging()
    assert PACKAGE.handlers == []
