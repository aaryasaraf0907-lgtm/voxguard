"""UI strings in English and Hindi, independent of Streamlit.

Every user-visible string in the UI goes through ``t(key, lang)``. Both dictionaries must have
identical keys and no empty values (enforced by tests).

Hindi (``hi``): reviewed by Aarya on PR #14 (CLAUDE.md section 11); her edits are applied. Strings
added after that review (phone-line simulation, 503 handling) still need her review.
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
    "simulate_help": (
        "The server passes the recording through the selected telephone codec, at the bitrate"
        " used in evaluation, before the detector scores it."
    ),
    "codec_label": "Phone codec",
    "codec_gsm": "GSM — not used in training",
    "codecs_error": "Could not load the codec list; the server's default codec is used.",
    "no_codecs": "This server's ffmpeg has none of the phone codecs, so simulation is unavailable.",
    "simulate_on": "Phone-line simulation is on: the next analysis uses the codec-degraded audio.",
    "simulate_off": "Phone-line simulation is off: the recording is analyzed as provided.",
    "result_clean": "Analyzed as provided (no phone-line simulation).",
    "result_simulated": "Analyzed after phone-line simulation with {codec}.",
    "result_header_clean": "Result: original audio",
    "result_header_phone": "Result: simulated phone line",
    "result_header": "Result",
    "flagged_banner": "Flagged: this recording scored at or above the threshold.",
    "not_flagged_banner": "Not flagged: this recording scored below the threshold.",
    "m_label": "Label",
    "m_score": "Fake score",
    "m_threshold": "Threshold",
    "m_latency": "Detector latency",
    "m_codec_ms": "Codec processing",
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
    "err_not_implemented": (
        "Phone-line simulation is not enabled on this server (VOXGUARD_SIMULATE_CODEC is not set)."
        " Turn the toggle off to analyze the original audio."
    ),
    "err_unavailable": (
        "Phone-line simulation isn't available on this server right now (its ffmpeg lacks the"
        " selected codec). Nothing was analyzed; turn the toggle off to analyze the original"
        " audio."
    ),
    "err_server": "The API reported an internal error. Please try again.",
    "err_bad_response": "The API returned an unexpected response.",
    "err_other": "The request failed.",
}

HI: dict[str, str] = {
    "app_title": "VoxGuard",
    "app_subtitle": ("जाँचें कि कोई आवाज़ रिकॉर्डिंग AI से बनी हो सकती है या नहीं, और चिह्नित कॉल देखें।"),
    "language": "Language / भाषा",
    "api_section": "API",
    "api_url": "पता: {url}",
    "status_connected": "जुड़ा हुआ",
    "status_detail": "डिटेक्टर: {detector} · मॉडल: {model}",
    "status_unavailable": "API उपलब्ध नहीं है",
    "stub_notice": (
        "स्टब डिटेक्टर चल रहा है: स्कोर केवल नमूना (प्लेसहोल्डर) मान हैं, असली पहचान के परिणाम नहीं।"
    ),
    "tab_analyze": "ऑडियो जाँचें",
    "tab_sessions": "चिह्नित कॉल",
    "tab_blocklist": "ब्लॉकलिस्ट",
    "tab_alerts": "अलर्ट",
    "analyze_header": "रिकॉर्डिंग जाँचें",
    "caller_label": "कॉलर आईडी (वैकल्पिक)",
    "caller_help": ("यदि कॉलर आईडी वाली रिकॉर्डिंग चिह्नित होती है, तो वह कॉलर ब्लॉकलिस्ट में जुड़ जाता है।"),
    "source_label": "ऑडियो स्रोत",
    "source_upload": "फ़ाइल अपलोड करें",
    "source_record": "माइक्रोफ़ोन से रिकॉर्ड करें",
    "upload_label": "ऑडियो फ़ाइल (WAV, FLAC, OGG या MP3)",
    "record_label": "आवाज़ का नमूना रिकॉर्ड करें",
    "no_audio": "जाँचने के लिए ऑडियो अपलोड या रिकॉर्ड करें।",
    "analyze_button": "जाँचें",
    "analyzing": "जाँच हो रही है…",
    "simulate_label": "फ़ोन लाइन का अनुकरण करें",
    "simulate_help": (
        "डिटेक्टर के स्कोर देने से पहले सर्वर रिकॉर्डिंग को चुने गए टेलीफ़ोन कोडेक से, मूल्यांकन"
        " वाले बिटरेट पर, गुज़ारता है।"
    ),
    "codec_label": "फ़ोन कोडेक",
    "codec_gsm": "GSM — प्रशिक्षण में उपयोग नहीं हुआ",
    "codecs_error": "API से कोडेक सूची नहीं मिल सकी; सर्वर का डिफ़ॉल्ट कोडेक इस्तेमाल होगा।",
    "no_codecs": "इस सर्वर के ffmpeg में कोई फ़ोन कोडेक नहीं है, इसलिए अनुकरण उपलब्ध नहीं है।",
    "simulate_on": "फ़ोन लाइन अनुकरण चालू है: अगली जाँच कोडेक से गुज़री रिकॉर्डिंग पर होगी।",
    "simulate_off": "फ़ोन लाइन अनुकरण बंद है: रिकॉर्डिंग जैसी है वैसी ही जाँची जाएगी।",
    "result_clean": "जैसी दी गई वैसी जाँची गई (फ़ोन लाइन अनुकरण नहीं)।",
    "result_simulated": "{codec} से फ़ोन लाइन अनुकरण के बाद जाँची गई।",
    "result_header_clean": "परिणाम: मूल ऑडियो",
    "result_header_phone": "परिणाम: फ़ोन लाइन अनुकरण के बाद",
    "result_header": "परिणाम",
    "flagged_banner": "चिह्नित: इस रिकॉर्डिंग का स्कोर सीमा के बराबर या उससे अधिक है।",
    "not_flagged_banner": "चिह्नित नहीं: इस रिकॉर्डिंग का स्कोर सीमा से कम है।",
    "m_label": "लेबल",
    "m_score": "नकली होने का स्कोर",
    "m_threshold": "सीमा",
    "m_latency": "डिटेक्टर प्रतिक्रिया समय",
    "m_codec_ms": "कोडेक प्रोसेसिंग समय",
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
    "sessions_header": "चिह्नित कॉल",
    "blocklist_header": "ब्लॉकलिस्ट",
    "alerts_header": "अलर्ट",
    "empty_sessions": "अभी तक कोई चिह्नित कॉल नहीं है।",
    "empty_blocklist": "ब्लॉकलिस्ट खाली है।",
    "empty_alerts": "अभी तक कोई अलर्ट नहीं है।",
    "col_time": "समय (UTC)",
    "col_caller": "कॉलर आईडी",
    "col_label": "लेबल",
    "col_score": "नकली होने का स्कोर",
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
    "err_not_implemented": (
        "इस सर्वर पर फ़ोन लाइन अनुकरण चालू नहीं है (VOXGUARD_SIMULATE_CODEC सेट नहीं है)।"
        " मूल ऑडियो जाँचने के लिए टॉगल बंद करें।"
    ),
    "err_unavailable": (
        "इस सर्वर पर अभी फ़ोन लाइन अनुकरण उपलब्ध नहीं है (उसके ffmpeg में चुना गया कोडेक नहीं है)।"
        " कुछ भी जाँचा नहीं गया; मूल ऑडियो जाँचने के लिए टॉगल बंद करें।"
    ),
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
