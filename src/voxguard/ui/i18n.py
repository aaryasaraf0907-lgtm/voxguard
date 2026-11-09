"""UI strings in English and Hindi, independent of Streamlit.

Every user-visible string in the UI goes through ``t(key, lang)``. Both dictionaries must have
identical keys and no empty values (enforced by tests).

NOTE: the Hindi (``hi``) strings are initial DRAFTS and need review by Aarya (CLAUDE.md section 11).
"""

from __future__ import annotations

LANGUAGES: dict[str, str] = {"en": "English", "hi": "हिन्दी"}
DEFAULT_LANGUAGE = "en"


class TranslationError(KeyError):
    """Unknown translation key or language."""


EN: dict[str, str] = {
    "app_title": "VoxGuard",
    "app_subtitle": (
        "Check whether a voice recording may be AI-generated, and review flagged calls."
    ),
    "language": "Language / भाषा",
    "hi_draft_notice": "Hindi text is a draft and has not been reviewed yet.",
    "api_section": "API",
    "api_url": "Address: {url}",
    "status_connected": "Connected",
    "status_detail": "Detector: {detector} · Model: {model}",
    "status_unavailable": "API unavailable",
    "stub_notice": (
        "The stub detector is running: scores are deterministic placeholders, "
        "not real detection results."
    ),
    "tab_analyze": "Analyze audio",
    "tab_sessions": "Flagged sessions",
    "tab_blocklist": "Blocklist",
    "tab_alerts": "Alerts",
    "analyze_header": "Analyze a recording",
    "caller_label": "Caller ID (optional)",
    "caller_help": "Flagged recordings with a caller ID add that caller to the blocklist.",
    "source_label": "Audio source",
    "source_upload": "Upload a file",
    "source_record": "Record with microphone",
    "upload_label": "Audio file (WAV, FLAC, OGG or MP3)",
    "record_label": "Record a voice sample",
    "no_audio": "Upload or record audio to analyze it.",
    "analyze_button": "Analyze",
    "analyzing": "Analyzing…",
    "simulate_label": "Simulate phone line",
    "simulate_unavailable": (
        "Not available yet: the API cannot apply codec simulation, "
        "so recordings are analyzed exactly as provided."
    ),
    "result_header": "Result",
    "flagged_banner": "Flagged: this recording scored at or above the threshold.",
    "not_flagged_banner": "Not flagged: this recording scored below the threshold.",
    "m_label": "Label",
    "m_score": "Fake score",
    "m_threshold": "Threshold",
    "m_latency": "Latency",
    "m_model": "Model version",
    "m_flagged": "Flagged",
    "m_codec": "Codec applied",
    "m_session": "Session ID: {session_id}",
    "label_real": "Real",
    "label_fake": "Fake",
    "yes": "Yes",
    "no": "No",
    "none_value": "None",
    "unit_ms": "{value} ms",
    "refresh": "Refresh",
    "utc_note": "All times are UTC.",
    "sessions_header": "Flagged sessions",
    "blocklist_header": "Blocklist",
    "alerts_header": "Alerts",
    "empty_sessions": "No flagged sessions yet.",
    "empty_blocklist": "The blocklist is empty.",
    "empty_alerts": "No alerts yet.",
    "col_time": "Time (UTC)",
    "col_caller": "Caller ID",
    "col_label": "Label",
    "col_score": "Fake score",
    "col_threshold": "Threshold",
    "col_model": "Model",
    "col_hits": "Flagged calls",
    "col_first_seen": "First seen (UTC)",
    "col_last_seen": "Last seen (UTC)",
    "col_message": "Message",
    "col_session": "Session ID",
    "missing_value": "—",
    "server_detail": "Server message: {detail}",
    "err_unreachable": (
        "Cannot reach the API. Check that the backend is running and the address is right."
    ),
    "err_timeout": "The API took too long to respond.",
    "err_too_large": "The file is too large for the API.",
    "err_unsupported": "The file could not be read as audio, or its format is not supported.",
    "err_invalid": (
        "The API rejected the request "
        "(for example, the audio is too long or has too many channels)."
    ),
    "err_not_implemented": "This feature is not available in the API yet.",
    "err_server": "The API reported an internal error. Please try again.",
    "err_bad_response": "The API returned an unexpected response.",
    "err_other": "The request failed.",
}

HI: dict[str, str] = {
    "app_title": "VoxGuard",
    "app_subtitle": ("जाँचें कि कोई आवाज़ रिकॉर्डिंग AI से बनी हो सकती है या नहीं, और चिह्नित कॉल देखें।"),
    "language": "Language / भाषा",
    "hi_draft_notice": "हिन्दी पाठ अभी प्रारूप है और इसकी समीक्षा नहीं हुई है।",
    "api_section": "API",
    "api_url": "पता: {url}",
    "status_connected": "जुड़ा हुआ",
    "status_detail": "डिटेक्टर: {detector} · मॉडल: {model}",
    "status_unavailable": "API उपलब्ध नहीं है",
    "stub_notice": ("स्टब डिटेक्टर चल रहा है: स्कोर निश्चित प्लेसहोल्डर हैं, असली पहचान के परिणाम नहीं।"),
    "tab_analyze": "ऑडियो जाँचें",
    "tab_sessions": "चिह्नित सत्र",
    "tab_blocklist": "ब्लॉकलिस्ट",
    "tab_alerts": "अलर्ट",
    "analyze_header": "रिकॉर्डिंग जाँचें",
    "caller_label": "कॉलर आईडी (वैकल्पिक)",
    "caller_help": "कॉलर आईडी के साथ चिह्नित रिकॉर्डिंग होने पर वह कॉलर ब्लॉकलिस्ट में जुड़ जाता है।",
    "source_label": "ऑडियो स्रोत",
    "source_upload": "फ़ाइल अपलोड करें",
    "source_record": "माइक्रोफ़ोन से रिकॉर्ड करें",
    "upload_label": "ऑडियो फ़ाइल (WAV, FLAC, OGG या MP3)",
    "record_label": "आवाज़ का नमूना रिकॉर्ड करें",
    "no_audio": "जाँचने के लिए ऑडियो अपलोड या रिकॉर्ड करें।",
    "analyze_button": "जाँचें",
    "analyzing": "जाँच हो रही है…",
    "simulate_label": "फ़ोन लाइन का अनुकरण करें",
    "simulate_unavailable": (
        "अभी उपलब्ध नहीं: API कोडेक अनुकरण नहीं कर सकता, इसलिए रिकॉर्डिंग जैसी है वैसी ही जाँची जाती है।"
    ),
    "result_header": "परिणाम",
    "flagged_banner": "चिह्नित: इस रिकॉर्डिंग का स्कोर सीमा के बराबर या उससे अधिक है।",
    "not_flagged_banner": "चिह्नित नहीं: इस रिकॉर्डिंग का स्कोर सीमा से कम है।",
    "m_label": "लेबल",
    "m_score": "नकली स्कोर",
    "m_threshold": "सीमा",
    "m_latency": "विलंब",
    "m_model": "मॉडल संस्करण",
    "m_flagged": "चिह्नित",
    "m_codec": "लागू कोडेक",
    "m_session": "सत्र आईडी: {session_id}",
    "label_real": "असली",
    "label_fake": "नकली",
    "yes": "हाँ",
    "no": "नहीं",
    "none_value": "कोई नहीं",
    "unit_ms": "{value} मि.से.",
    "refresh": "ताज़ा करें",
    "utc_note": "सभी समय UTC में हैं।",
    "sessions_header": "चिह्नित सत्र",
    "blocklist_header": "ब्लॉकलिस्ट",
    "alerts_header": "अलर्ट",
    "empty_sessions": "अभी तक कोई चिह्नित सत्र नहीं है।",
    "empty_blocklist": "ब्लॉकलिस्ट खाली है।",
    "empty_alerts": "अभी तक कोई अलर्ट नहीं है।",
    "col_time": "समय (UTC)",
    "col_caller": "कॉलर आईडी",
    "col_label": "लेबल",
    "col_score": "नकली स्कोर",
    "col_threshold": "सीमा",
    "col_model": "मॉडल",
    "col_hits": "चिह्नित कॉल",
    "col_first_seen": "पहली बार देखा (UTC)",
    "col_last_seen": "आख़िरी बार देखा (UTC)",
    "col_message": "संदेश",
    "col_session": "सत्र आईडी",
    "missing_value": "—",
    "server_detail": "सर्वर का संदेश: {detail}",
    "err_unreachable": "API से संपर्क नहीं हो सका। जाँचें कि बैकएंड चल रहा है और पता सही है।",
    "err_timeout": "API ने उत्तर देने में बहुत समय लिया।",
    "err_too_large": "फ़ाइल API के लिए बहुत बड़ी है।",
    "err_unsupported": "फ़ाइल को ऑडियो के रूप में पढ़ा नहीं जा सका, या उसका प्रारूप समर्थित नहीं है।",
    "err_invalid": "API ने अनुरोध अस्वीकार कर दिया (जैसे ऑडियो बहुत लंबा है या उसमें बहुत अधिक चैनल हैं)।",
    "err_not_implemented": "यह सुविधा अभी API में उपलब्ध नहीं है।",
    "err_server": "API ने आंतरिक त्रुटि बताई। कृपया फिर कोशिश करें।",
    "err_bad_response": "API से अप्रत्याशित उत्तर मिला।",
    "err_other": "अनुरोध विफल रहा।",
}

TRANSLATIONS: dict[str, dict[str, str]] = {"en": EN, "hi": HI}


def t(key: str, lang: str = DEFAULT_LANGUAGE, **fmt: object) -> str:
    """Look up ``key`` in ``lang``; ``fmt`` fills ``{placeholders}``. Unknown key/lang raises."""
    try:
        table = TRANSLATIONS[lang]
    except KeyError:
        raise TranslationError(
            f"unknown language {lang!r}; known: {sorted(TRANSLATIONS)}"
        ) from None
    try:
        text = table[key]
    except KeyError:
        raise TranslationError(f"unknown translation key {key!r} for language {lang!r}") from None
    return text.format(**fmt) if fmt else text
