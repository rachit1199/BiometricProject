# Multimodal Face + Voice Authentication for Secure Mobile Banking Login (v2)

Phone-friendly web app: **face recognition + voice verification**, quality-aware score fusion, anti-replay challenge, liveness checks, and **OTP fallback** when confidence is low.

## Run
```bash
pip install -r requirements.txt
python app.py                  # http://localhost:5000
HTTPS=1 python app.py          # required for camera/mic from a phone on your network (accept the cert warning)
ENROLL_INVITE=mycode python app.py   # optionally restrict who can enrol
python -m pytest -q tests      # 12 tests
```

## What's new in v2
| Area | Upgrade |
|---|---|
| Anti-replay | Server-issued random **digit challenge** (single-use, 120 s expiry, bound to the username). **Replay detector** rejects a login whose voice features are near-identical to a previous login. Optional **ASR** (`ENABLE_ASR`, faster-whisper) checks the spoken digits. |
| Liveness | Adds **head-movement and blink cues** (Haar eyes) on top of motion + sharpness; optional experimental screen-moire check. Failed checks are reported by name and audited. |
| Quality | Face (sharpness, size) and voice (SNR, clipping, length) quality scores. Bad captures are rejected with a helpful message, and **adaptive fusion** down-weights the lower-quality modality. |
| Risk | Grant threshold rises after recent failed attempts (`RISK_STEP`), pushing borderline logins to OTP. |
| Enrolment | **Consistency checks**: face frames must agree with each other (blocks two people enrolling as one), and so must the 3 voice samples. Optional invite code. |
| Sessions | Real session tokens (hashed at rest, 15 min expiry), `/api/session`, `/api/logout`, `/api/audit`, `DELETE /api/account` (wipes biometric templates). |
| Storage | Lockouts, OTPs (stored as HMAC, never plaintext), sessions, challenges and audit trail now persist in SQLite, so they survive restarts. Templates stay Fernet-encrypted. |
| API hardening | Per-IP rate limit, security headers/CSP, no user enumeration (challenge issued for any name), OTP endpoint now respects lockout. |
| Evaluation | `evaluate.py` adds bootstrap 95% CIs, FRR@FAR=1%, FAR@FRR=1%, a check of your current `config.py` policy (grant/OTP/reject rates for genuine vs impostor), `roc.csv`, `report.json`, and `--plot` for a DET curve. |
| UI | New `templates/index.html`: challenge digits on screen, countdown, progress bar, OTP box, session timer, activity log, sign-out and delete-my-data. |

## Pipeline
| Stage | File | What it does |
|---|---|---|
| Capture | `templates/index.html` | 10 frames + 3.5 s audio (16 kHz WAV encoded in the browser) |
| Face | `face.py` | Haar detect, crop, equalise, LBPH distance -> 0-1 score |
| Liveness | `face.py` | Motion, sharpness, head movement / blink, optional spectral-spike test |
| Voice | `voice.py` | Trim, 19 MFCC + deltas (76-dim mean/std), z-distance to voiceprint -> score |
| Challenge / ASR | `store.py`, `speech.py` | Random digits, single-use; optional transcription check |
| Fusion | `fusion.py` | `S = w_f*S_face + w_v*S_voice` with quality-adjusted weights; grant / OTP / reject, risk-adjusted |
| State | `store.py` | Encrypted templates, SQLite lockout/OTP/session/audit |
| API | `app.py` | `/api/challenge /enroll /login /otp /session /logout /audit /account /health` |
| Evaluation | `evaluate.py` | FAR / FRR / EER, CIs, DET, best weights |

All thresholds are in `config.py` (every liveness/quality/replay value is an **untuned starting point**).

## Evaluate with real data
```
dataset/<person>/sample_01.jpg + sample_01.wav ...
python evaluate.py dataset --enroll 3 --plot
```
Copy the suggested `W_FACE`, `W_VOICE`, `T_HIGH` into `config.py`. Aim for 10+ people and 8+ samples each; report the CI, not just the EER.

## Known limitations (state these in your report)
- Thresholds are untuned. Calibrate on real recordings, including noisy and low-light conditions.
- Liveness is still basic. Haar blink/eye detection is noisy, a held photo moves with the hand, and a video replayed on a screen can pass. Upgrade: landmark-based blink/head-turn challenge or a trained presentation-attack detector.
- The replay detector only catches near-identical copies. A fresh recording replayed through a speaker, or synthetic/cloned voice, needs ASR (`ENABLE_ASR`) plus an anti-spoofing model (e.g. AASIST).
- LBPH + MFCC statistics are lightweight classical methods. For higher accuracy put ArcFace (face) and ECAPA-TDNN (voice) behind the same `match()` functions.
- Demo mode returns the OTP in the API response; set `DEMO_MODE = False` and send by SMS/email. Rate limiting is in-memory per process, and sessions have no real bank backend.
- Biometric data is sensitive personal data: get consent, and use the delete endpoint and retention limits appropriate to your jurisdiction.
