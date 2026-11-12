"""Flask server. Endpoints: /api/challenge /enroll /login /otp /session /logout /audit /account(DELETE) /health"""
import os, time
from collections import defaultdict, deque
from flask import Flask, jsonify, render_template, request
import config as cfg
import face, voice, fusion, store, speech

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024
app.config["TEMPLATES_AUTO_RELOAD"] = True   # pick up edits to templates/index.html without restarting
_hits = defaultdict(deque)


def err(msg, code=400, **extra):
    return jsonify({"ok": False, "error": msg, **extra}), code


def ip():
    return request.remote_addr or "?"


def bearer():
    h = request.headers.get("Authorization", "")
    return store.session_user(h[7:]) if h.startswith("Bearer ") else None


@app.before_request
def rate_limit():
    if request.method == "POST" and request.path.startswith("/api/"):
        dq, now = _hits[ip()], time.time()
        while dq and now - dq[0] > 60:
            dq.popleft()
        if len(dq) >= cfg.RATE_LIMIT_PER_MIN:
            return err("too many requests", 429)
        dq.append(now)


@app.after_request
def headers(r):
    r.headers.update({"X-Content-Type-Options": "nosniff", "X-Frame-Options": "DENY", "Cache-Control": "no-store",
                      "Referrer-Policy": "no-referrer",
                      "Permissions-Policy": "camera=(self), microphone=(self)",
                      "Content-Security-Policy": "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; img-src 'self' data:"})
    return r


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/health")
def health():
    return jsonify({"ok": True, "challenge": cfg.REQUIRE_CHALLENGE, "asr": cfg.ENABLE_ASR, "demo": cfg.DEMO_MODE,
                    "invite_required": bool(cfg.ENROLL_INVITE_CODE)})


@app.post("/api/challenge")
def challenge():
    """Issued for any username (known or not) so the endpoint does not reveal which accounts exist."""
    user = (request.get_json(force=True).get("username") or "").strip()
    cid, digits = store.issue_challenge(user)
    return jsonify({"ok": True, "challenge_id": cid, "digits": digits, "ttl": cfg.CHALLENGE_TTL_SECONDS})


@app.post("/api/enroll")
def enroll():
    j = request.get_json(force=True)
    user = (j.get("username") or "").strip()
    if cfg.ENROLL_INVITE_CODE and j.get("invite") != cfg.ENROLL_INVITE_CODE:
        return err("invalid invite code", 403)
    if not store.valid_username(user):
        return err("username must be 3-32 chars: letters, digits, _ . -")
    if store.exists(user):
        return err("user already enrolled", 409)
    faces = face.extract_faces([face.decode_image(f) for f in j.get("frames", [])])
    if len(faces) < cfg.FACE_ENROLL_MIN_FRAMES:
        return err(f"need at least {cfg.FACE_ENROLL_MIN_FRAMES} clear face frames, got {len(faces)}")
    if face.self_consistency(faces) > cfg.FACE_ENROLL_MAX_DIST:
        return err("face frames look inconsistent (more than one person, or heavy motion blur). Retry alone, facing the camera")
    audios = [voice.decode_wav(a) for a in j.get("audios", [])]
    if len(audios) < 3:
        return err("need at least 3 voice samples")
    feats = []
    for i, y in enumerate(audios, 1):
        ok, why = voice.check_quality(y)
        if not ok:
            return err(f"voice sample {i}: {why}")
        feats.append(voice.features(y))
    if voice.enroll_consistency(feats) > cfg.VOICE_ENROLL_MAX_DIST:
        return err("voice samples are inconsistent, record all three yourself in the same place")
    vp = voice.build_voiceprint(feats)
    store.save(user, "face", faces=faces)
    store.save(user, "voice", mu=vp["mu"], sd=vp["sd"])
    store.audit(user, "enroll", ip(), faces=len(faces), voices=len(feats))
    return jsonify({"ok": True, "message": f"{user} enrolled", "face_frames": len(faces), "voice_samples": len(feats)})


@app.post("/api/login")
def login():
    j = request.get_json(force=True)
    user = (j.get("username") or "").strip()
    if not store.valid_username(user) or not store.exists(user):
        return err("authentication failed", 401)
    if store.locked(user):
        store.audit(user, "blocked_locked", ip())
        return err("account temporarily locked, try later or contact the bank", 423)

    def fail(msg, code=401, event="fail", **extra):
        store.register_failure(user)
        store.audit(user, event, ip(), reason=msg, **extra)
        return err(msg, code, **extra)

    digits = None
    if cfg.REQUIRE_CHALLENGE:
        digits = store.consume_challenge(user, j.get("challenge_id"))
        if digits is None:
            return err("challenge missing or expired, request a new one", 400)

    infos = face.analyze_all([face.decode_image(f) for f in j.get("frames", [])])
    if len(infos) < cfg.FACE_MIN_FRAMES:
        return fail("no clear face detected")
    live, live_info = face.liveness(infos)
    if not live:
        return fail("liveness check failed", event="liveness_fail", liveness=live_info)
    try:
        y = voice.decode_wav(j.get("audio", ""))
    except Exception:
        return err("could not read audio")
    ok, why = voice.check_quality(y)
    if not ok:
        return err(why)
    if digits and speech.verify_digits(y, digits) is False:
        return fail("spoken digits did not match the challenge", event="digits_mismatch")

    vp = store.load(user, "voice")
    feat = voice.features(y)
    if voice.is_replay(feat, store.past_voice_features(user), vp["sd"]):
        return fail("replayed audio detected", event="replay")

    faces = [i["crop"] for i in infos]
    s_face, _ = face.match(list(store.load(user, "face")["faces"]), faces)
    s_voice, _ = voice.match(vp, y, feat)
    q_face, q_voice = face.quality(infos), voice.quality(y)["q"]
    fails = store.recent_failures(user)
    decision, fused = fusion.decide(s_face, s_voice, q_face, q_voice, fails)
    wf, wv = fusion.weights(q_face, q_voice)
    scores = {"face": round(s_face, 3), "voice": round(s_voice, 3), "fused": round(fused, 3),
              "quality": {"face": round(q_face, 2), "voice": round(q_voice, 2)}, "weights": {"face": round(wf, 2), "voice": round(wv, 2)}}
    store.audit(user, f"decision_{decision}", ip(), scores=scores, liveness=live_info)

    if decision == fusion.GRANT:
        store.clear_failures(user)
        store.log_voice(user, feat)
        return jsonify({"ok": True, "decision": "grant", "scores": scores, "token": store.issue_session(user), "expires_in": cfg.SESSION_TTL_SECONDS})
    if decision == fusion.OTP:
        store.log_voice(user, feat)
        out = {"ok": True, "decision": "otp", "scores": scores, "message": "Confidence is low. Enter the OTP sent to you."}
        if cfg.DEMO_MODE:
            out["demo_otp"] = store.issue_otp(user)
        else:
            store.issue_otp(user)  # deliver via SMS/email here
        return jsonify(out)
    store.register_failure(user)
    return jsonify({"ok": False, "decision": "reject", "scores": scores, "error": "authentication failed"}), 401


@app.post("/api/otp")
def otp():
    j = request.get_json(force=True)
    user = (j.get("username") or "").strip()
    if store.locked(user):
        return err("account temporarily locked", 423)
    if store.check_otp(user, j.get("code", "")):
        store.clear_failures(user)
        store.audit(user, "otp_ok", ip())
        return jsonify({"ok": True, "decision": "grant", "token": store.issue_session(user), "expires_in": cfg.SESSION_TTL_SECONDS})
    store.register_failure(user)
    store.audit(user, "otp_fail", ip())
    return err("invalid or expired OTP", 401)


@app.get("/api/session")
def session():
    u = bearer()
    return jsonify({"ok": True, "user": u}) if u else err("not signed in", 401)


@app.post("/api/logout")
def logout():
    h = request.headers.get("Authorization", "")
    u = bearer()
    store.end_session(h[7:])
    if u:
        store.audit(u, "logout", ip())
    return jsonify({"ok": True})


@app.get("/api/audit")
def audit_log():
    u = bearer()
    if not u:
        return err("not signed in", 401)
    n = min(100, max(1, request.args.get("limit", 50, type=int)))
    return jsonify({"ok": True, "events": store.recent_audit(u, n)})


@app.get("/api/account")
def account():
    """Account overview for the signed-in user: template info, session, counters and last activity."""
    u = bearer()
    if not u:
        return err("not signed in", 401)
    tok = request.headers["Authorization"][7:]
    ev = store.recent_audit(u, 500)
    count = lambda *names: sum(e["event"] in names for e in ev)
    last_ok = next((e["ts"] for e in ev[1:] if e["event"] in ("decision_grant", "otp_ok")), None)  # skip current login
    face_t = store.load(u, "face")
    return jsonify({
        "ok": True, "user": u, "enrolled_at": store.enrolled_at(u),
        "face_frames": int(len(face_t["faces"])) if face_t else 0,
        "session_expires_at": store.session_expiry(tok), "server_time": time.time(),
        "failed_attempts": store.recent_failures(u), "max_attempts": cfg.MAX_FAILED_ATTEMPTS, "locked": store.locked(u),
        "otp_enabled": True, "challenge": cfg.REQUIRE_CHALLENGE, "asr": cfg.ENABLE_ASR,
        "thresholds": {"grant": cfg.T_HIGH, "otp": cfg.T_LOW},
        "totals": {"signins": count("decision_grant", "otp_ok"), "otp_required": count("decision_otp"),
                   "failures": count("fail", "liveness_fail", "digits_mismatch", "replay", "decision_reject", "otp_fail"),
                   "blocked": count("blocked_locked")},
        "last_signin": last_ok, "this_ip": ip(), "session_ttl": cfg.SESSION_TTL_SECONDS,
        "policy": {"lockout_seconds": cfg.LOCKOUT_SECONDS, "max_attempts": cfg.MAX_FAILED_ATTEMPTS, "otp_ttl": cfg.OTP_TTL_SECONDS,
                   "otp_max_tries": cfg.OTP_MAX_TRIES, "session_ttl": cfg.SESSION_TTL_SECONDS, "rate_limit": cfg.RATE_LIMIT_PER_MIN,
                   "challenge_ttl": cfg.CHALLENGE_TTL_SECONDS, "challenge_digits": cfg.CHALLENGE_DIGITS,
                   "min_snr": cfg.VOICE_MIN_SNR_DB, "max_clipping": cfg.VOICE_MAX_CLIPPING, "min_voice_seconds": cfg.VOICE_MIN_SECONDS,
                   "replay_eps": cfg.VOICE_REPLAY_EPS, "adaptive_fusion": cfg.ADAPTIVE_FUSION, "risk_step": cfg.RISK_STEP,
                   "min_modality": cfg.MIN_MODALITY_SCORE, "w_face": cfg.W_FACE, "w_voice": cfg.W_VOICE,
                   "live_min_motion": cfg.LIVE_MIN_MOTION, "live_max_motion": cfg.LIVE_MAX_MOTION, "live_min_sharp": cfg.LIVE_MIN_SHARPNESS,
                   "live_min_head": cfg.LIVE_MIN_HEAD_MOVE, "require_active_cue": cfg.REQUIRE_ACTIVE_CUE,
                   "invite_required": bool(cfg.ENROLL_INVITE_CODE), "demo_mode": cfg.DEMO_MODE}})


@app.delete("/api/account")
def delete_account():
    u = bearer()
    if not u:
        return err("not signed in", 401)
    store.audit(u, "account_deleted", ip())
    store.delete_user(u)
    return jsonify({"ok": True, "message": "biometric templates deleted"})


if __name__ == "__main__":
    ctx = "adhoc" if os.environ.get("HTTPS") == "1" else None
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), ssl_context=ctx)
