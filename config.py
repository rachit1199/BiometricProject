"""All tunable settings in one place. Re-tune thresholds on your own data (see evaluate.py)."""
import os

DATA_DIR = os.environ.get("BIO_DATA_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# ---- Face ----
FACE_SIZE = 100
FACE_MIN_FRAMES = 4
FACE_DIST_MID = 65.0       # LBPH distance mapped to score 0.5
FACE_DIST_SLOPE = 8.0
FACE_ENROLL_MIN_FRAMES = 12
FACE_ENROLL_MAX_DIST = 90.0  # enrolment frames must agree with each other (rejects 2 different people)

# ---- Face liveness (passive + light active cues; all thresholds UNTUNED, calibrate on real data) ----
LIVE_MIN_MOTION = 1.2      # mean abs pixel change between consecutive face crops
LIVE_MAX_MOTION = 40.0     # above this: frame swapping / jumping
LIVE_MIN_SHARPNESS = 40.0  # Laplacian variance of the face crop
REQUIRE_ACTIVE_CUE = True  # need head movement OR a blink across the clip
LIVE_MIN_HEAD_MOVE = 0.004 # std of face-centre position, relative to face width
LIVE_USE_MOIRE = False     # experimental: spectral-spike test for screen replay
LIVE_MAX_SPECTRAL_SPIKE = 60.0

# ---- Voice ----
SAMPLE_RATE = 16000
VOICE_MIN_SECONDS = 1.5
VOICE_MIN_RMS = 0.005
VOICE_MAX_CLIPPING = 0.02  # fraction of samples at full scale
VOICE_MIN_SNR_DB = 8.0
VOICE_DIST_MID = 2.0
VOICE_DIST_SLOPE = 0.5
VOICE_ENROLL_MAX_DIST = 3.0   # enrolment recordings must agree with each other
VOICE_REPLAY_EPS = 0.05       # login too close to a past login = replayed copy (natural repeats differ)
VOICE_REPLAY_HISTORY = 10

# ---- Challenge-response (anti-replay) ----
REQUIRE_CHALLENGE = True   # server issues random digits; single-use, expires
CHALLENGE_DIGITS = 6
CHALLENGE_TTL_SECONDS = 120
ENABLE_ASR = False         # True + `pip install faster-whisper`: check the spoken digits match the challenge
ASR_MODEL = "tiny.en"

# ---- Fusion ----
W_FACE = 0.5
W_VOICE = 0.5
ADAPTIVE_FUSION = True     # down-weight a modality whose capture quality is poor
MIN_QUALITY_WEIGHT = 0.3
T_HIGH = 0.75
T_LOW = 0.50
MIN_MODALITY_SCORE = 0.30
RISK_STEP = 0.05           # T_HIGH rises by this per recent failed attempt (max RISK_MAX)
RISK_MAX = 0.10

# ---- Account / API protection ----
MAX_FAILED_ATTEMPTS = 3
LOCKOUT_SECONDS = 300
OTP_TTL_SECONDS = 300
OTP_MAX_TRIES = 3
SESSION_TTL_SECONDS = 900
RATE_LIMIT_PER_MIN = 30    # per IP, POST /api/*
ENROLL_INVITE_CODE = os.environ.get("ENROLL_INVITE", "")  # if set, enrolment needs this code
DEMO_MODE = True           # True: OTP is returned in the API response. Set False in production.
