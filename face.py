"""Face pipeline: detect -> crop -> LBPH match, with passive + light active liveness and capture-quality scoring."""
import base64
import cv2
import numpy as np
import config as cfg

_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
_eyes = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")


def decode_image(data_url: str):
    b64 = data_url.split(",", 1)[1] if "," in data_url else data_url
    arr = np.frombuffer(base64.b64decode(b64), np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def analyze(img):
    """Return dict(crop, cx, cy, w, eyes) for the largest face, or None."""
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    faces = _cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    H, W = gray.shape
    roi = gray[y:y + int(h * 0.6), x:x + w]
    eyes = _eyes.detectMultiScale(roi, 1.1, 6, minSize=(max(8, w // 10),) * 2)
    crop = cv2.equalizeHist(cv2.resize(gray[y:y + h, x:x + w], (cfg.FACE_SIZE, cfg.FACE_SIZE)))
    return {"crop": crop, "cx": (x + w / 2) / W, "cy": (y + h / 2) / H, "w": w / W, "eyes": len(eyes)}


def analyze_all(images):
    return [a for a in (analyze(i) for i in images) if a]


def extract_face(img):
    a = analyze(img)
    return a["crop"] if a else None


def extract_faces(images):
    return [a["crop"] for a in analyze_all(images)]


def _spectral_spike(crop):
    f = np.fft.fftshift(np.abs(np.fft.fft2(crop.astype(np.float32))))
    h, w = f.shape
    yy, xx = np.ogrid[:h, :w]
    hf = f[np.hypot(yy - h / 2, xx - w / 2) > 0.35 * min(h, w)]
    return float(hf.max() / (hf.mean() + 1e-9))


def _blinked(eyes):
    """Eyes seen, then lost for 1-3 frames, then seen again."""
    s = "".join("1" if e >= 1 else "0" for e in eyes)
    return any(f"1{'0' * k}1" in s for k in (1, 2, 3))


def liveness(infos):
    """Liveness on per-frame infos from analyze_all(). Returns (ok, details). Basic, NOT a full PAD system."""
    if len(infos) < 3:
        return False, {"reason": "too few face frames"}
    crops = [i["crop"] for i in infos]
    motion = float(np.mean([np.mean(cv2.absdiff(a, b)) for a, b in zip(crops, crops[1:])]))
    sharp = float(np.mean([cv2.Laplacian(c, cv2.CV_64F).var() for c in crops]))
    head = float((np.std([i["cx"] for i in infos]) + np.std([i["cy"] for i in infos])) / np.mean([i["w"] for i in infos]))
    blink = _blinked([i["eyes"] for i in infos])
    spike = _spectral_spike(crops[len(crops) // 2])
    checks = {
        "motion": cfg.LIVE_MIN_MOTION <= motion <= cfg.LIVE_MAX_MOTION,
        "sharpness": sharp >= cfg.LIVE_MIN_SHARPNESS,
        "active_cue": (head >= cfg.LIVE_MIN_HEAD_MOVE or blink) if cfg.REQUIRE_ACTIVE_CUE else True,
        "screen": (spike <= cfg.LIVE_MAX_SPECTRAL_SPIKE) if cfg.LIVE_USE_MOIRE else True,
    }
    failed = [k for k, v in checks.items() if not v]
    return not failed, {"motion": round(motion, 2), "sharpness": round(sharp, 1), "head_move": round(head, 4),
                        "blink": blink, "spike": round(spike, 1), "failed": failed}


def quality(infos):
    """Capture quality in [0.2, 1] from sharpness and face size; used to weight fusion."""
    if not infos:
        return 0.2
    sharp = np.mean([cv2.Laplacian(i["crop"], cv2.CV_64F).var() for i in infos])
    size = np.mean([i["w"] for i in infos])
    return float(max(0.2, 0.5 * min(1, sharp / 150) + 0.5 * min(1, size / 0.3)))


def _recognizer(enrolled):
    rec = cv2.face.LBPHFaceRecognizer_create(radius=2, neighbors=8, grid_x=8, grid_y=8)
    rec.train(list(enrolled), np.zeros(len(enrolled), dtype=np.int32))
    return rec


def self_consistency(faces):
    """Train on even frames, test odd frames. Returns median distance (low = same person throughout)."""
    rec = _recognizer(faces[0::2])
    return float(np.median([rec.predict(f)[1] for f in faces[1::2]]))


def distance_to_score(d):
    return float(1.0 / (1.0 + np.exp((d - cfg.FACE_DIST_MID) / cfg.FACE_DIST_SLOPE)))


def match(enrolled, probe_faces):
    rec = _recognizer(enrolled)
    d = float(np.median([rec.predict(f)[1] for f in probe_faces]))
    return distance_to_score(d), d
