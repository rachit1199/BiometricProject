"""Quality-aware score-level fusion and a three-way, risk-adjusted decision."""
import config as cfg

GRANT, OTP, REJECT = "grant", "otp", "reject"


def weights(q_face=1.0, q_voice=1.0):
    if not cfg.ADAPTIVE_FUSION:
        q_face = q_voice = 1.0
    wf = cfg.W_FACE * max(q_face, cfg.MIN_QUALITY_WEIGHT)
    wv = cfg.W_VOICE * max(q_voice, cfg.MIN_QUALITY_WEIGHT)
    t = wf + wv
    return wf / t, wv / t


def fuse(s_face, s_voice, q_face=1.0, q_voice=1.0):
    wf, wv = weights(q_face, q_voice)
    return wf * s_face + wv * s_voice


def risk_bump(recent_failures: int) -> float:
    return min(cfg.RISK_MAX, cfg.RISK_STEP * max(0, recent_failures))


def decide(s_face, s_voice, q_face=1.0, q_voice=1.0, recent_failures=0):
    """Return (decision, fused_score). Either modality very weak rejects outright."""
    s = fuse(s_face, s_voice, q_face, q_voice)
    if min(s_face, s_voice) < cfg.MIN_MODALITY_SCORE:
        return REJECT, s
    if s >= cfg.T_HIGH + risk_bump(recent_failures):
        return GRANT, s
    if s >= cfg.T_LOW:
        return OTP, s
    return REJECT, s
