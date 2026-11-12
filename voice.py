"""Voice pipeline: WAV -> MFCC statistics -> normalised distance to voiceprint, with quality + replay checks."""
import io, base64
import numpy as np
import librosa
import soundfile as sf
import config as cfg


def decode_wav(b64: str):
    raw = base64.b64decode(b64.split(",", 1)[1] if "," in b64 else b64)
    y, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != cfg.SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=sr, target_sr=cfg.SAMPLE_RATE)
    return y


def snr_db(y):
    """Rough SNR: loudest 10% of 20 ms frames vs quietest 10%."""
    n = int(0.02 * cfg.SAMPLE_RATE)
    if len(y) < 10 * n:
        return 0.0
    e = np.sort((y[: len(y) // n * n].reshape(-1, n) ** 2).mean(axis=1))
    k = max(1, len(e) // 10)
    return float(10 * np.log10((e[-k:].mean() + 1e-12) / (e[:k].mean() + 1e-12)))


def quality(y):
    """Return dict(seconds, rms, clipping, snr_db, q) where q in [0.2, 1] weights fusion."""
    t, _ = librosa.effects.trim(y, top_db=30)
    clip = float(np.mean(np.abs(y) >= 0.99)) if len(y) else 0.0
    snr = snr_db(y)
    q = max(0.2, min(1.0, (snr - 5) / 25)) * (1 - min(0.8, clip * 20))
    return {"seconds": len(t) / cfg.SAMPLE_RATE, "rms": float(np.sqrt(np.mean(t ** 2))) if len(t) else 0.0,
            "clipping": clip, "snr_db": snr, "q": float(max(0.2, q))}


def check_quality(y):
    """Return (ok, reason)."""
    info = quality(y)
    if info["seconds"] < cfg.VOICE_MIN_SECONDS:
        return False, "speech too short, speak for at least 2 seconds"
    if info["rms"] < cfg.VOICE_MIN_RMS:
        return False, "audio too quiet"
    if info["clipping"] > cfg.VOICE_MAX_CLIPPING:
        return False, "audio is clipping, move the phone away or speak softer"
    if info["snr_db"] < cfg.VOICE_MIN_SNR_DB:
        return False, "too much background noise, find a quieter place"
    return True, ""


def features(y) -> np.ndarray:
    """19 MFCC (c0 dropped) + deltas -> mean and std over time (76-dim vector)."""
    y, _ = librosa.effects.trim(y, top_db=30)
    y = librosa.effects.preemphasis(y)
    m = librosa.feature.mfcc(y=y, sr=cfg.SAMPLE_RATE, n_mfcc=20, n_fft=512, hop_length=160, n_mels=40)[1:]
    stack = np.vstack([m, librosa.feature.delta(m)])
    return np.concatenate([stack.mean(axis=1), stack.std(axis=1)]).astype(np.float64)


def build_voiceprint(feature_list):
    f = np.vstack(feature_list)
    mu = f.mean(axis=0)
    sd = f.std(axis=0, ddof=1) if len(f) > 1 else np.ones_like(mu)
    pooled = float(np.median(sd))
    sd = np.sqrt(0.5 * sd ** 2 + 0.5 * pooled ** 2) + 1e-6
    return {"mu": mu, "sd": sd}


def enroll_consistency(feature_list):
    """Worst leave-one-out z-distance between an enrolment sample and the others (low = same speaker)."""
    f = np.vstack(feature_list)
    sd = build_voiceprint(feature_list)["sd"]
    return float(max(np.mean(np.abs((f[i] - np.delete(f, i, 0).mean(0)) / sd)) for i in range(len(f))))


def is_replay(feat, past, sd):
    """A login almost identical to a previous one is a copy: real repetitions never match this closely."""
    return any(float(np.mean(np.abs((feat - p) / sd))) < cfg.VOICE_REPLAY_EPS for p in past)


def distance_to_score(d):
    return float(1.0 / (1.0 + np.exp((d - cfg.VOICE_DIST_MID) / cfg.VOICE_DIST_SLOPE)))


def match(voiceprint, y, feat=None):
    feat = features(y) if feat is None else feat
    d = float(np.mean(np.abs((feat - voiceprint["mu"]) / voiceprint["sd"])))
    return distance_to_score(d), d
