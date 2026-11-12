"""Optional spoken-digit check (ASR). Off by default; enable with cfg.ENABLE_ASR and faster-whisper."""
import re
import config as cfg

_WORDS = {"zero": "0", "oh": "0", "o": "0", "one": "1", "two": "2", "to": "2", "too": "2", "three": "3",
          "four": "4", "for": "4", "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9"}
_model = None


def _lev(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def text_to_digits(text: str) -> str:
    out = []
    for tok in re.findall(r"[a-z]+|\d", text.lower()):
        out.append(_WORDS.get(tok, tok if tok.isdigit() else ""))
    return "".join(out)


def transcribe(y):
    global _model
    from faster_whisper import WhisperModel
    if _model is None:
        _model = WhisperModel(cfg.ASR_MODEL, compute_type="int8")
    segs, _ = _model.transcribe(y, language="en", beam_size=1)
    return " ".join(s.text for s in segs)


def verify_digits(y, expected: str):
    """True/False, or None when ASR is disabled/unavailable (caller must not treat None as a pass)."""
    if not cfg.ENABLE_ASR:
        return None
    try:
        heard = text_to_digits(transcribe(y))
    except Exception:
        return None
    return _lev(heard, expected) <= 1
