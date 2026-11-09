import ast
from pathlib import Path
from typing import get_args

import pytest

from voxguard.ui import i18n
from voxguard.ui.api_client import ErrorKind
from voxguard.ui.i18n import EN, HI, TranslationError, t

APP = Path(__file__).parents[1] / "src" / "voxguard" / "ui" / "app.py"


def test_same_keys_in_both_languages() -> None:
    assert set(EN) == set(HI)


@pytest.mark.parametrize("lang", ["en", "hi"])
def test_no_empty_values(lang: str) -> None:
    assert all(v.strip() for v in i18n.TRANSLATIONS[lang].values())


def test_placeholders_match_between_languages() -> None:
    from string import Formatter

    def names(s: str) -> set[str]:
        return {f for _, f, _, _ in Formatter().parse(s) if f}

    assert {k: names(v) for k, v in EN.items()} == {k: names(v) for k, v in HI.items()}


def test_hindi_is_actually_translated() -> None:
    same = {k for k in EN if EN[k] == HI[k]}
    # brand name, "API" heading, bilingual selector label, em dash: intentionally identical
    assert same <= {"app_title", "api_section", "language", "missing_value"}


def test_unknown_key_fails_clearly() -> None:
    with pytest.raises(TranslationError, match="no_such_key"):
        t("no_such_key", "en")


def test_unknown_language_fails_clearly() -> None:
    with pytest.raises(TranslationError, match="fr"):
        t("app_title", "fr")


def test_formatting() -> None:
    assert "http://x" in t("api_url", "en", url="http://x")
    assert "http://x" in t("api_url", "hi", url="http://x")


def _t_call_keys() -> set[str]:
    keys = set()
    for node in ast.walk(ast.parse(APP.read_text())):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "t"
            and node.args
            and isinstance(node.args[0], ast.Constant)
        ):
            keys.add(node.args[0].value)
    return keys


def test_every_literal_key_used_by_the_ui_exists() -> None:
    keys = _t_call_keys()
    assert len(keys) > 20  # the scan actually found the UI's keys
    assert keys <= set(EN)


def test_dynamic_keys_used_by_the_ui_exist() -> None:
    dynamic = {f"err_{k}" for k in get_args(ErrorKind)}
    dynamic |= {"label_real", "label_fake", "source_upload", "source_record"}
    assert dynamic <= set(EN)


def test_no_unused_translation_keys() -> None:
    dynamic = {f"err_{k}" for k in get_args(ErrorKind)}
    dynamic |= {"label_real", "label_fake", "source_upload", "source_record"}
    constants = {
        n.value
        for n in ast.walk(ast.parse(APP.read_text()))
        if isinstance(n, ast.Constant) and isinstance(n.value, str)
    }  # keys also reach t() through variables, e.g. the tab and header tuples
    assert set(EN) - constants - dynamic == set()


def test_ui_has_no_hardcoded_display_strings() -> None:
    display = {
        "title", "header", "subheader", "write", "caption", "info", "warning", "error",
        "success", "button", "text_input", "radio", "file_uploader", "audio_input",
        "toggle", "metric", "markdown", "spinner", "selectbox", "tabs",
    }  # fmt: skip
    offenders = []
    for node in ast.walk(ast.parse(APP.read_text())):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in display
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            offenders.append((node.func.attr, node.args[0].value))
    assert offenders == []
