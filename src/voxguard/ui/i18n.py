"""UI strings in English and Hindi, independent of Streamlit.

Every user-visible string in the UI goes through ``t(key, lang)``. Both dictionaries must have
identical keys and no empty values (enforced by tests).

Hindi (``hi``): reviewed by Aarya on PR #14 (CLAUDE.md section 11); her edits are applied. Strings
added after that review (phone-line simulation, 503 handling, server busy, the
timeline) still need her review.
"""

from __future__ import annotations

LANGUAGES: dict[str, str] = {"en": "English", "hi": "हिन्दी"}
DEFAULT_LANGUAGE = "en"


class TranslationError(KeyError):
    """Unknown translation key or language."""


EN: dict[str, str] = {
    "app_title": "LineTrue",
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
    "upload_label": "Audio file (WAV, FLAC, OGG, MP3 or M4A)",
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
    "caution_near_threshold": (
        "Close to the decision threshold: this result is uncertain. Treat it as a hint, not a"
        " verdict."
    ),
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
    "err_decoder_unavailable": (
        "This server can't decode this file format right now (its ffmpeg is missing)."
        " Nothing was analyzed; try a WAV or FLAC file."
    ),
    "timeline_label": "Show live-monitoring timeline",
    "timeline_help": (
        "Scores the recording window by window (about 4 s each), as a live call monitor would,"
        " and shows when the running average reached the threshold. Same verdict as without it;"
        " a little slower."
    ),
    "timeline_header": "Live-monitoring timeline",
    "timeline_flagged_at": (
        "A live monitor using this threshold would have flagged this call after {seconds} s of"
        " audio."
    ),
    "timeline_never": "The running average never reached the threshold.",
    "timeline_flagged_briefly": (
        "The running average reached the threshold after {seconds} s of audio, then fell below"
        " it: a live monitor would have raised an early alert that the full recording does not"
        " support."
    ),
    "timeline_note": (
        "Computed window by window on this recording; not a measured early-detection rate."
    ),
    "timeline_x": "Seconds of audio",
    "timeline_window": "Window score",
    "timeline_running": "Running average",
    "timeline_threshold": "Threshold",
    "timeline_replay": "Replay",
    "report_title": "LineTrue incident report",
    "report_download": "Download report (HTML)",
    "report_print": "Print / Save as PDF",
    "report_evidence": "Evidence trail",
    "report_evidence_ok": (
        "Intact: all {entries} entries of the tamper-evident log check out against the records."
    ),
    "report_evidence_problem": (
        "Problem: the tamper-evident log does not match the records. Treat this report with care."
    ),
    "report_evidence_head": "Chain head now",
    "report_evidence_note": (
        "Each record is chained by SHA-256 to the one before it. Keep this head, or the one in"
        " the alert webhook, to compare later; re-check at {url}. Tamper-evident, not proof of"
        " who did what."
    ),
    "report_kind": "Record",
    "report_kind_session": "Session (verdict)",
    "report_kind_windows": "Per-window scores",
    "report_kind_decision": "Transfer decision",
    "report_kind_event": "Transfer step",
    "report_hash": "Hash",
    "report_generated": "Generated {time}. Times are UTC.",
    "report_flagged": "Flagged: the call scored at or above the threshold.",
    "report_not_flagged": "Not flagged: the call scored below the threshold.",
    "report_session": "Session",
    "report_time": "Analysed at",
    "report_source": "Source",
    "report_source_live": "Live call monitor",
    "report_source_upload": "Uploaded recording",
    "report_timeline": "Risk during the call",
    "report_first_over": (
        "The running average first reached the threshold after {seconds} s of audio."
    ),
    "report_never_over": "The running average never reached the threshold.",
    "report_no_timeline": (
        "No per-window scores: this was an uploaded recording, scored as a whole."
    ),
    "report_transfers": "Transfers checked against this call",
    "report_no_transfers": "No transfer was checked against this call.",
    "report_privacy": "No audio was kept: LineTrue stores scores, thresholds and decisions only.",
    "report_disclaimer": (
        "A detection score is evidence to verify, not proof. Confirm through a call-back on a"
        " known number before acting."
    ),
    "tx_amount": "Amount",
    "tx_beneficiary": "Beneficiary",
    "tx_tier": "Risk",
    "tx_action": "Action",
    "tx_status": "Status",
    "tx_steps": "Checks done",
    "tx_tier_unknown": "Not enough audio yet",
    "tx_tier_low": "Low",
    "tx_tier_caution": "Caution",
    "tx_tier_high": "High",
    "tx_action_allow": "Allow",
    "tx_action_step_up": "Verify first",
    "tx_action_hold": "Hold",
    "tx_status_allowed": "Allowed",
    "tx_status_pending": "Waiting for verification",
    "tx_status_held": "Held",
    "tx_status_released": "Released after verification",
    "tx_method_callback": "Call-back on a known number",
    "tx_method_otp": "OTP",
    "tx_method_second_approver": "Second approver",
    "tx_reason_risk_unknown": "Not enough of the call has been heard yet.",
    "tx_reason_risk_low": "The call's risk is below the threshold.",
    "tx_reason_risk_caution": "The call's risk is just over the threshold.",
    "tx_reason_risk_high": "The call's risk is well over the threshold.",
    "tx_reason_not_enough_audio": "Fewer than two 4 s windows have been scored.",
    "tx_reason_caller_flagged_before": "This caller was flagged on an earlier call.",
    "tx_reason_high_value": "The amount is at or above the high-value limit.",
    "monitor_tx_title": "Transfer approval during this call",
    "monitor_tx_help": (
        "Try to approve a transfer while the call is going on: LineTrue checks the call's risk"
        " first."
    ),
    "monitor_tx_amount": "Amount (₹)",
    "monitor_tx_beneficiary": "Beneficiary (label)",
    "monitor_tx_approve": "Approve transfer",
    "monitor_tx_need_call": "Start a call first.",
    "monitor_tx_allowed": "Allowed: no extra check is needed at this call's risk.",
    "monitor_tx_step_up": (
        "Verify first: complete any one step below before the transfer goes ahead."
    ),
    "monitor_tx_held": (
        "Held: this transfer needs a second approver and a call-back on a known number."
    ),
    "monitor_tx_released": "Released after verification.",
    "monitor_tx_verifier": "Your name (who did the check; a hold needs two different people)",
    "monitor_tx_mark": "Mark done",
    "monitor_tx_note": (
        "Steps are recorded as done by the person named; this demo sends no OTP and places no call."
    ),
    "monitor_report": "Open the incident report",
    "monitor_report_download": "Download the report",
    "tab_transactions": "Transactions",
    "transactions_header": "Transfer decisions",
    "empty_transactions": "No transfer has been checked yet.",
    "col_report": "Report",
    "col_amount": "Amount (₹)",
    "monitor_link": "Open the live call monitor",
    "monitor_link_help": "Listens while a call is going on and warns during the call.",
    "monitor_title": "LineTrue live call monitor",
    "monitor_intro": "Listens while the call is going on and updates the risk every 2 seconds.",
    "monitor_note": (
        "Listens through this computer's microphone, or plays a recording; it is not connected"
        " to a phone line. The first score needs about 4 s of audio."
    ),
    "monitor_consent": (
        "Before you start: tell the other person that the call is being checked for AI-generated"
        " voices. Use it only on calls you take part in."
    ),
    "monitor_guidance_title": "What to do now",
    "monitor_guidance_callback": (
        "Hang up and call back on a number you already have for this person or organisation."
    ),
    "monitor_guidance_question": (
        "Ask something only the real person would know. Urgency or secrecy is not a reason to skip"
        " checks."
    ),
    "monitor_guidance_report": (
        "If money, OTPs or passwords were asked for, report it: call 1930 (National Cyber Crime"
        " Helpline) or use cybercrime.gov.in."
    ),
    "monitor_caller": "Caller ID (optional)",
    "monitor_codec": "Phone line",
    "monitor_codec_none": "None (audio as captured)",
    "monitor_start_mic": "Start listening (microphone)",
    "monitor_file": "Or play a recording and monitor it",
    "monitor_play": "Play and monitor",
    "monitor_stop": "Stop",
    "monitor_status_idle": "Not listening.",
    "monitor_status_connecting": "Connecting…",
    "monitor_status_listening": "Listening… the first score comes after about 4 s.",
    "monitor_safe": "Below the threshold so far.",
    "monitor_checking": "Checking: one window is over the threshold; waiting for more audio.",
    "monitor_caution": "Caution: the risk is just over the threshold.",
    "monitor_elevated": (
        "Slightly above the threshold, not yet at the alert level: keep listening. Genuine calls"
        " often sit here."
    ),
    "monitor_caution_advice": (
        "Verify the caller another way before acting on this call: call back on a number you"
        " already trust."
    ),
    "monitor_alert": "Warning: this voice may be AI-generated.",
    "monitor_alert_advice": (
        "Do not approve payments or share codes on this call. Hang up and call back on a number"
        " you already trust."
    ),
    "monitor_alert_cleared": (
        "The risk reached the alert level at {seconds} s and is below the threshold now. Verify"
        " the caller before acting."
    ),
    "monitor_elapsed": "Audio",
    "monitor_window": "Window score",
    "monitor_running": "Running average",
    "monitor_threshold": "Threshold",
    "monitor_seconds": "Seconds of audio",
    "monitor_busy": "Server busy: a window was skipped.",
    "monitor_final_flagged": (
        "Call ended: flagged (score {score} ≥ threshold {threshold}). Saved as session {session}."
    ),
    "monitor_final_clear": (
        "Call ended: not flagged (score {score} < threshold {threshold}). Saved as session"
        " {session}."
    ),
    "monitor_final_none": "The call ended before any audio could be scored.",
    "monitor_final_no_live_alert": (
        "It was flagged on the whole call, but never reached the live alert level (or was too"
        " short for one)."
    ),
    "monitor_error": "Error: {detail}",
    "monitor_mic_denied": (
        "Microphone access was blocked. Allow it in the browser's address bar and try again."
    ),
    "monitor_unsupported": (
        "This browser cannot capture audio here. Use a recent Chrome, Edge or Firefox, on"
        " localhost or HTTPS."
    ),
    "monitor_disconnected": "The connection to the API closed.",
    "timeline_series": "Series",
    "err_busy": (
        "The server is busy with other analyses right now. Nothing was analyzed; please try"
        " again in a few seconds."
    ),
    "err_server": "The API reported an internal error. Please try again.",
    "err_bad_response": "The API returned an unexpected response.",
    "err_other": "The request failed.",
}

HI: dict[str, str] = {
    "app_title": "LineTrue",
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
    "upload_label": "ऑडियो फ़ाइल (WAV, FLAC, OGG, MP3 या M4A)",
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
    "caution_near_threshold": (
        "यह परिणाम निर्णय-सीमा के क़रीब है, इसलिए अनिश्चित है। इसे संकेत समझें, अंतिम निर्णय नहीं।"
    ),
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
    "err_unsupported": ("फ़ाइल को ऑडियो के रूप में पढ़ा नहीं जा सका, या उसका प्रारूप समर्थित नहीं है।"),
    "err_invalid": ("API ने अनुरोध अस्वीकार कर दिया (जैसे ऑडियो बहुत लंबा है या उसमें बहुत अधिक चैनल हैं)।"),
    "err_not_implemented": (
        "इस सर्वर पर फ़ोन लाइन अनुकरण चालू नहीं है (VOXGUARD_SIMULATE_CODEC सेट नहीं है)।"
        " मूल ऑडियो जाँचने के लिए टॉगल बंद करें।"
    ),
    "err_unavailable": (
        "इस सर्वर पर अभी फ़ोन लाइन अनुकरण उपलब्ध नहीं है (उसके ffmpeg में चुना गया कोडेक नहीं है)।"
        " कुछ भी जाँचा नहीं गया; मूल ऑडियो जाँचने के लिए टॉगल बंद करें।"
    ),
    "err_decoder_unavailable": (
        "यह सर्वर अभी इस फ़ाइल प्रारूप को डिकोड नहीं कर सकता (उसमें ffmpeg नहीं है)।"
        " कुछ भी जाँचा नहीं गया; WAV या FLAC फ़ाइल आज़माएँ।"
    ),
    "timeline_label": "लाइव निगरानी टाइमलाइन दिखाएँ",
    "timeline_help": (
        "रिकॉर्डिंग को विंडो दर विंडो (लगभग 4 सेकंड की) जाँचता है, जैसे लाइव कॉल मॉनिटर करता,"
        " और दिखाता है कि चालू औसत सीमा तक कब पहुँचा। नतीजा वही रहता है; थोड़ा धीमा।"
    ),
    "timeline_header": "लाइव निगरानी टाइमलाइन",
    "timeline_flagged_at": (
        "इस सीमा पर एक लाइव मॉनिटर इस कॉल को ऑडियो के {seconds} सेकंड बाद चिह्नित कर देता।"
    ),
    "timeline_never": "चालू औसत कभी सीमा तक नहीं पहुँचा।",
    "timeline_flagged_briefly": (
        "चालू औसत ऑडियो के {seconds} सेकंड बाद सीमा तक पहुँचा, फिर उससे नीचे आ गया: लाइव मॉनिटर"
        " एक शुरुआती चेतावनी देता, जिसका पूरी रिकॉर्डिंग समर्थन नहीं करती।"
    ),
    "timeline_note": "इस रिकॉर्डिंग पर विंडो दर विंडो गणना; यह मापी गई शुरुआती पहचान दर नहीं है।",
    "timeline_x": "ऑडियो के सेकंड",
    "timeline_window": "विंडो स्कोर",
    "timeline_running": "चालू औसत",
    "timeline_threshold": "सीमा",
    "timeline_replay": "दोबारा चलाएँ",
    "report_title": "LineTrue घटना रिपोर्ट",
    "report_download": "रिपोर्ट डाउनलोड करें (HTML)",
    "report_print": "प्रिंट करें / PDF के रूप में सहेजें",
    "report_evidence": "साक्ष्य रिकॉर्ड",
    "report_evidence_ok": (
        "सुरक्षित: छेड़छाड़-पहचान लॉग की सभी {entries} प्रविष्टियाँ रिकॉर्ड से मेल खाती हैं।"
    ),
    "report_evidence_problem": (
        "समस्या: छेड़छाड़-पहचान लॉग रिकॉर्ड से मेल नहीं खाता। इस रिपोर्ट को सावधानी से देखें।"
    ),
    "report_evidence_head": "अभी की श्रृंखला का शीर्ष",
    "report_evidence_note": (
        "हर रिकॉर्ड SHA-256 से पिछले रिकॉर्ड से जुड़ा है। बाद में मिलान के लिए यह शीर्ष, या अलर्ट"
        " वेबहुक वाला शीर्ष, रखें; {url} पर दोबारा जाँचें। यह छेड़छाड़ पकड़ता है, यह सबूत नहीं कि"
        " किसने क्या किया।"
    ),
    "report_kind": "रिकॉर्ड",
    "report_kind_session": "सत्र (निर्णय)",
    "report_kind_windows": "विंडो-वार स्कोर",
    "report_kind_decision": "ट्रांसफ़र निर्णय",
    "report_kind_event": "ट्रांसफ़र कदम",
    "report_hash": "हैश",
    "report_generated": "{time} पर बनी। समय UTC में हैं।",
    "report_flagged": "चिह्नित: कॉल का स्कोर सीमा के बराबर या उससे अधिक रहा।",
    "report_not_flagged": "चिह्नित नहीं: कॉल का स्कोर सीमा से कम रहा।",
    "report_session": "सत्र",
    "report_time": "जाँच का समय",
    "report_source": "स्रोत",
    "report_source_live": "लाइव कॉल मॉनिटर",
    "report_source_upload": "अपलोड की गई रिकॉर्डिंग",
    "report_timeline": "कॉल के दौरान जोखिम",
    "report_first_over": "चालू औसत ऑडियो के {seconds} सेकंड बाद पहली बार सीमा तक पहुँचा।",
    "report_never_over": "चालू औसत कभी सीमा तक नहीं पहुँचा।",
    "report_no_timeline": (
        "विंडो-वार स्कोर नहीं: यह अपलोड की गई रिकॉर्डिंग थी, जिसे पूरा एक साथ जाँचा गया।"
    ),
    "report_transfers": "इस कॉल के आधार पर जाँचे गए ट्रांसफ़र",
    "report_no_transfers": "इस कॉल के आधार पर कोई ट्रांसफ़र नहीं जाँचा गया।",
    "report_privacy": "कोई ऑडियो नहीं रखा गया: LineTrue केवल स्कोर, सीमाएँ और निर्णय सहेजता है।",
    "report_disclaimer": (
        "पहचान स्कोर पुष्टि करने लायक संकेत है, सबूत नहीं। कुछ करने से पहले किसी ज्ञात नंबर पर"
        " वापस कॉल करके पुष्टि करें।"
    ),
    "tx_amount": "राशि",
    "tx_beneficiary": "लाभार्थी",
    "tx_tier": "जोखिम",
    "tx_action": "कार्रवाई",
    "tx_status": "स्थिति",
    "tx_steps": "पूरी हुई जाँच",
    "tx_tier_unknown": "अभी पर्याप्त ऑडियो नहीं",
    "tx_tier_low": "कम",
    "tx_tier_caution": "सावधानी",
    "tx_tier_high": "अधिक",
    "tx_action_allow": "अनुमति",
    "tx_action_step_up": "पहले पुष्टि करें",
    "tx_action_hold": "रोकें",
    "tx_status_allowed": "अनुमति दी गई",
    "tx_status_pending": "पुष्टि की प्रतीक्षा",
    "tx_status_held": "रोका गया",
    "tx_status_released": "पुष्टि के बाद जारी",
    "tx_method_callback": "ज्ञात नंबर पर वापस कॉल",
    "tx_method_otp": "OTP",
    "tx_method_second_approver": "दूसरा अनुमोदक",
    "tx_reason_risk_unknown": "अभी कॉल का पर्याप्त हिस्सा नहीं सुना गया है।",
    "tx_reason_risk_low": "कॉल का जोखिम सीमा से कम है।",
    "tx_reason_risk_caution": "कॉल का जोखिम सीमा से थोड़ा ऊपर है।",
    "tx_reason_risk_high": "कॉल का जोखिम सीमा से काफ़ी ऊपर है।",
    "tx_reason_not_enough_audio": "अभी 4 सेकंड की दो से कम विंडो का स्कोर बना है।",
    "tx_reason_caller_flagged_before": "यह कॉलर पिछली किसी कॉल पर चिह्नित हुआ था।",
    "tx_reason_high_value": "राशि उच्च-मूल्य सीमा के बराबर या उससे अधिक है।",
    "monitor_tx_title": "इस कॉल के दौरान ट्रांसफ़र की स्वीकृति",
    "monitor_tx_help": ("कॉल चलते समय ट्रांसफ़र स्वीकृत करके देखें: LineTrue पहले कॉल का जोखिम जाँचता है।"),
    "monitor_tx_amount": "राशि (₹)",
    "monitor_tx_beneficiary": "लाभार्थी (लेबल)",
    "monitor_tx_approve": "ट्रांसफ़र स्वीकृत करें",
    "monitor_tx_need_call": "पहले कॉल शुरू करें।",
    "monitor_tx_allowed": "अनुमति: इस कॉल के जोखिम पर किसी अतिरिक्त जाँच की ज़रूरत नहीं।",
    "monitor_tx_step_up": "पहले पुष्टि करें: ट्रांसफ़र से पहले नीचे का कोई एक कदम पूरा करें।",
    "monitor_tx_held": ("रोका गया: इस ट्रांसफ़र के लिए दूसरा अनुमोदक और ज्ञात नंबर पर वापस कॉल चाहिए।"),
    "monitor_tx_released": "पुष्टि के बाद जारी।",
    "monitor_tx_verifier": "आपका नाम (जाँच किसने की; रोके गए ट्रांसफ़र के लिए दो अलग व्यक्ति चाहिए)",
    "monitor_tx_mark": "पूरा हुआ चिह्नित करें",
    "monitor_tx_note": ("कदम नामित व्यक्ति द्वारा पूरे दर्ज होते हैं; यह डेमो न OTP भेजता है न कॉल करता है।"),
    "monitor_report": "घटना रिपोर्ट खोलें",
    "monitor_report_download": "रिपोर्ट डाउनलोड करें",
    "tab_transactions": "लेन-देन",
    "transactions_header": "ट्रांसफ़र निर्णय",
    "empty_transactions": "अभी तक कोई ट्रांसफ़र नहीं जाँचा गया।",
    "col_report": "रिपोर्ट",
    "col_amount": "राशि (₹)",
    "monitor_link": "लाइव कॉल मॉनिटर खोलें",
    "monitor_link_help": "कॉल चलते समय सुनता है और कॉल के दौरान ही चेतावनी देता है।",
    "monitor_title": "LineTrue लाइव कॉल मॉनिटर",
    "monitor_intro": "कॉल चलते समय सुनता है और हर 2 सेकंड में जोखिम अपडेट करता है।",
    "monitor_note": (
        "इस कंप्यूटर के माइक्रोफ़ोन से सुनता है, या रिकॉर्डिंग चलाता है; यह फ़ोन लाइन से जुड़ा"
        " नहीं है। पहले स्कोर के लिए लगभग 4 सेकंड का ऑडियो चाहिए।"
    ),
    "monitor_consent": (
        "शुरू करने से पहले: सामने वाले व्यक्ति को बताएँ कि कॉल की AI से बनी आवाज़ों के लिए जाँच हो"
        " रही है। इसे केवल उन्हीं कॉल पर इस्तेमाल करें जिनमें आप शामिल हैं।"
    ),
    "monitor_guidance_title": "अब क्या करें",
    "monitor_guidance_callback": (
        "कॉल काटें और इस व्यक्ति या संस्था के उस नंबर पर वापस कॉल करें जो आपके पास पहले से है।"
    ),
    "monitor_guidance_question": (
        "कुछ ऐसा पूछें जो केवल असली व्यक्ति जानता हो। जल्दबाज़ी या गोपनीयता जाँच छोड़ने का कारण नहीं है।"
    ),
    "monitor_guidance_report": (
        "अगर पैसे, OTP या पासवर्ड माँगे गए हों तो रिपोर्ट करें: 1930 (राष्ट्रीय साइबर अपराध हेल्पलाइन)"
        " पर कॉल करें या cybercrime.gov.in पर जाएँ।"
    ),
    "monitor_caller": "कॉलर आईडी (वैकल्पिक)",
    "monitor_codec": "फ़ोन लाइन",
    "monitor_codec_none": "कोई नहीं (जैसा रिकॉर्ड हुआ)",
    "monitor_start_mic": "सुनना शुरू करें (माइक्रोफ़ोन)",
    "monitor_file": "या कोई रिकॉर्डिंग चलाएँ और उसकी निगरानी करें",
    "monitor_play": "चलाएँ और निगरानी करें",
    "monitor_stop": "रोकें",
    "monitor_status_idle": "सुन नहीं रहा।",
    "monitor_status_connecting": "जुड़ रहा है…",
    "monitor_status_listening": "सुन रहा है… पहला स्कोर लगभग 4 सेकंड बाद आएगा।",
    "monitor_safe": "अब तक सीमा से नीचे।",
    "monitor_checking": "जाँच जारी: एक विंडो सीमा से ऊपर है; और ऑडियो की प्रतीक्षा है।",
    "monitor_caution": "सावधानी: जोखिम सीमा से थोड़ा ऊपर है।",
    "monitor_elevated": (
        "सीमा से थोड़ा ऊपर, पर अभी चेतावनी के स्तर तक नहीं: सुनते रहें। असली कॉल अक्सर यहीं रहती हैं।"
    ),
    "monitor_caution_advice": (
        "इस कॉल पर कुछ भी करने से पहले कॉलर की किसी और तरीके से पुष्टि करें: किसी भरोसेमंद नंबर पर वापस कॉल करें।"
    ),
    "monitor_alert": "चेतावनी: यह आवाज़ AI से बनी हो सकती है।",
    "monitor_alert_advice": (
        "इस कॉल पर कोई भुगतान स्वीकृत न करें और कोई कोड न बताएँ। कॉल काटें और किसी भरोसेमंद"
        " नंबर पर वापस कॉल करें।"
    ),
    "monitor_alert_cleared": (
        "जोखिम {seconds} सेकंड पर चेतावनी के स्तर तक पहुँचा था और अभी सीमा से नीचे है। कुछ भी करने से पहले"
        " कॉलर की पुष्टि करें।"
    ),
    "monitor_elapsed": "ऑडियो",
    "monitor_window": "विंडो स्कोर",
    "monitor_running": "चालू औसत",
    "monitor_threshold": "सीमा",
    "monitor_seconds": "ऑडियो के सेकंड",
    "monitor_busy": "सर्वर व्यस्त: एक विंडो छोड़ दी गई।",
    "monitor_final_flagged": (
        "कॉल समाप्त: चिह्नित (स्कोर {score} ≥ सीमा {threshold})। सत्र {session} के रूप में सहेजा गया।"
    ),
    "monitor_final_clear": (
        "कॉल समाप्त: चिह्नित नहीं (स्कोर {score} < सीमा {threshold})। सत्र {session} के रूप में सहेजा गया।"
    ),
    "monitor_final_none": "किसी भी ऑडियो का स्कोर बनने से पहले कॉल समाप्त हो गई।",
    "monitor_final_no_live_alert": (
        "पूरी कॉल पर यह चिह्नित हुई, पर लाइव चेतावनी के स्तर तक कभी नहीं पहुँची (या उसके लिए बहुत छोटी थी)।"
    ),
    "monitor_error": "त्रुटि: {detail}",
    "monitor_mic_denied": (
        "माइक्रोफ़ोन की अनुमति रोकी गई। ब्राउज़र के पता बार में अनुमति दें और फिर कोशिश करें।"
    ),
    "monitor_unsupported": (
        "यह ब्राउज़र यहाँ ऑडियो नहीं ले सकता। localhost या HTTPS पर हाल का Chrome, Edge या"
        " Firefox इस्तेमाल करें।"
    ),
    "monitor_disconnected": "API से कनेक्शन बंद हो गया।",
    "timeline_series": "श्रृंखला",
    "err_busy": (
        "सर्वर अभी दूसरे विश्लेषणों में व्यस्त है। कुछ भी जाँचा नहीं गया; कृपया कुछ सेकंड बाद फिर कोशिश करें।"
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
