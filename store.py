"""Encrypted templates (Fernet) + SQLite state: lockouts, hashed OTPs, sessions, challenges, replay log, audit trail."""
import hashlib, hmac, io, json, os, re, secrets, sqlite3, time
import numpy as np
from cryptography.fernet import Fernet
import config as cfg

os.makedirs(os.path.join(cfg.DATA_DIR, "users"), exist_ok=True)
_KEY_PATH = os.path.join(cfg.DATA_DIR, "secret.key")
if not os.path.exists(_KEY_PATH):
    with open(_KEY_PATH, "wb") as f:
        f.write(Fernet.generate_key())
    os.chmod(_KEY_PATH, 0o600)
_KEY = open(_KEY_PATH, "rb").read()
_fernet = Fernet(_KEY)
_DB = os.path.join(cfg.DATA_DIR, "state.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS attempts(user TEXT PRIMARY KEY, fails INT, locked_until REAL);
CREATE TABLE IF NOT EXISTS otps(user TEXT PRIMARY KEY, h TEXT, expires REAL, tries INT);
CREATE TABLE IF NOT EXISTS sessions(h TEXT PRIMARY KEY, user TEXT, expires REAL);
CREATE TABLE IF NOT EXISTS challenges(id TEXT PRIMARY KEY, user TEXT, digits TEXT, expires REAL);
CREATE TABLE IF NOT EXISTS voice_log(user TEXT, ts REAL, feat TEXT);
CREATE TABLE IF NOT EXISTS audit(ts REAL, user TEXT, event TEXT, ip TEXT, detail TEXT);
"""


def _q(sql, args=(), one=False, many=False):
    c = sqlite3.connect(_DB, timeout=10)
    try:
        c.row_factory = sqlite3.Row
        cur = c.execute(sql, args)
        res = cur.fetchone() if one else cur.fetchall() if many else None
        c.commit()
        return res
    finally:
        c.close()


_c = sqlite3.connect(_DB); _c.executescript(_SCHEMA); _c.commit(); _c.close()


def _h(user, secret):
    return hmac.new(_KEY, f"{user}:{secret}".encode(), hashlib.sha256).hexdigest()


def valid_username(name):
    return bool(re.fullmatch(r"[A-Za-z0-9_.-]{3,32}", name or ""))


# ---- templates ----
def _path(user, kind):
    return os.path.join(cfg.DATA_DIR, "users", f"{user}.{kind}.enc")


def save(user, kind, **arrays):
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    with open(_path(user, kind), "wb") as f:
        f.write(_fernet.encrypt(buf.getvalue()))


def load(user, kind):
    p = _path(user, kind)
    if not os.path.exists(p):
        return None
    with open(p, "rb") as f:
        data = np.load(io.BytesIO(_fernet.decrypt(f.read())))
    return {k: data[k] for k in data.files}


def exists(user):
    return os.path.exists(_path(user, "face")) and os.path.exists(_path(user, "voice"))


def delete_user(user):
    for k in ("face", "voice"):
        if os.path.exists(_path(user, k)):
            os.remove(_path(user, k))
    for t in ("attempts", "otps", "voice_log"):
        _q(f"DELETE FROM {t} WHERE user=?", (user,))
    _q("DELETE FROM sessions WHERE user=?", (user,))


# ---- lockout ----
def locked(user):
    r = _q("SELECT locked_until FROM attempts WHERE user=?", (user,), one=True)
    return bool(r and r["locked_until"] > time.time())


def recent_failures(user):
    r = _q("SELECT fails FROM attempts WHERE user=?", (user,), one=True)
    return r["fails"] if r else 0


def register_failure(user):
    r = _q("SELECT fails FROM attempts WHERE user=?", (user,), one=True)
    fails = (r["fails"] if r else 0) + 1
    until = 0.0
    if fails >= cfg.MAX_FAILED_ATTEMPTS:
        until, fails = time.time() + cfg.LOCKOUT_SECONDS, 0
    _q("INSERT OR REPLACE INTO attempts VALUES(?,?,?)", (user, fails, until))


def clear_failures(user):
    _q("DELETE FROM attempts WHERE user=?", (user,))


# ---- OTP (stored as HMAC, single-use) ----
def issue_otp(user):
    code = f"{secrets.randbelow(10**6):06d}"
    _q("INSERT OR REPLACE INTO otps VALUES(?,?,?,0)", (user, _h(user, code), time.time() + cfg.OTP_TTL_SECONDS))
    if cfg.DEMO_MODE:
        print(f"[OTP] {user}: {code}  (demo mode: replace with SMS/email delivery)")
    return code


def check_otp(user, code):
    o = _q("SELECT * FROM otps WHERE user=?", (user,), one=True)
    if not o or o["expires"] < time.time() or o["tries"] >= cfg.OTP_MAX_TRIES:
        _q("DELETE FROM otps WHERE user=?", (user,))
        return False
    _q("UPDATE otps SET tries=tries+1 WHERE user=?", (user,))
    if hmac.compare_digest(o["h"], _h(user, str(code))):
        _q("DELETE FROM otps WHERE user=?", (user,))
        return True
    return False


# ---- challenge (spoken digits, single-use) ----
def issue_challenge(user):
    cid = secrets.token_urlsafe(12)
    digits = "".join(str(secrets.randbelow(10)) for _ in range(cfg.CHALLENGE_DIGITS))
    _q("DELETE FROM challenges WHERE expires<?", (time.time(),))
    _q("INSERT INTO challenges VALUES(?,?,?,?)", (cid, user, digits, time.time() + cfg.CHALLENGE_TTL_SECONDS))
    return cid, digits


def consume_challenge(user, cid):
    r = _q("SELECT * FROM challenges WHERE id=?", (cid or "",), one=True)
    _q("DELETE FROM challenges WHERE id=?", (cid or "",))
    if r and r["user"] == user and r["expires"] >= time.time():
        return r["digits"]
    return None


# ---- sessions ----
def issue_session(user):
    tok = secrets.token_urlsafe(32)
    _q("INSERT INTO sessions VALUES(?,?,?)", (_h("sess", tok), user, time.time() + cfg.SESSION_TTL_SECONDS))
    return tok


def session_user(tok):
    r = _q("SELECT user, expires FROM sessions WHERE h=?", (_h("sess", tok or ""),), one=True)
    return r["user"] if r and r["expires"] >= time.time() else None


def session_expiry(tok):
    r = _q("SELECT expires FROM sessions WHERE h=?", (_h("sess", tok or ""),), one=True)
    return r["expires"] if r else None


def enrolled_at(user):
    p = _path(user, "face")
    return os.path.getmtime(p) if os.path.exists(p) else None


def end_session(tok):
    _q("DELETE FROM sessions WHERE h=?", (_h("sess", tok or ""),))


# ---- voice replay log ----
def past_voice_features(user):
    rows = _q("SELECT feat FROM voice_log WHERE user=? ORDER BY ts DESC LIMIT ?", (user, cfg.VOICE_REPLAY_HISTORY), many=True)
    return [np.array(json.loads(r["feat"])) for r in rows]


def log_voice(user, feat):
    _q("INSERT INTO voice_log VALUES(?,?,?)", (user, time.time(), json.dumps([float(x) for x in feat])))


# ---- audit ----
def audit(user, event, ip="", **detail):
    _q("INSERT INTO audit VALUES(?,?,?,?,?)", (time.time(), user, event, ip, json.dumps(detail)))


def recent_audit(user, n=20):
    rows = _q("SELECT ts,event,ip,detail FROM audit WHERE user=? ORDER BY ts DESC LIMIT ?", (user, n), many=True)
    return [{"ts": r["ts"], "event": r["event"], "ip": r["ip"], "detail": json.loads(r["detail"])} for r in rows]
