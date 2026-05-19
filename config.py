import os
from dotenv import load_dotenv
load_dotenv()

BACKEND_WEBHOOK_URL = os.environ.get("BACKEND_WEBHOOK_URL", "http://localhost:5000/api/ai/pipeline/results")
SHARED_SECRET      = os.environ.get("SHARED_SECRET", "dev-secret")


OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

# ── MCQ Timing ────────────────────────────────────────────────────
MCQ_INTERVAL_SECONDS: int = int(os.getenv("MCQ_INTERVAL_SECONDS", "1200"))

# ── Audio Recording ───────────────────────────────────────────────
AUDIO_SAMPLE_RATE: int    = 16000
AUDIO_CHANNELS: int       = 1
AUDIO_CHUNK_FRAMES: int   = 1024
AUDIO_RECORD_SECONDS: int = 30

# ── GPT Model ─────────────────────────────────────────────────────
GPT_MODEL: str = "gpt-4o"

# ── Transcription Model ───────────────────────────────────────────
WHISPER_MODEL: str = "whisper-1"

# ── Intelligent Memory ────────────────────────────────────────────
MEMORY_MAX_CHUNKS: int              = int(os.getenv("MEMORY_MAX_CHUNKS", "20"))
MEMORY_SUMMARY_TRIGGER: int         = int(os.getenv("MEMORY_SUMMARY_TRIGGER", "15"))
MEMORY_RECENT_CONTEXT_CHUNKS: int   = int(os.getenv("MEMORY_RECENT_CONTEXT_CHUNKS", "5"))
MEMORY_IMPORTANCE_THRESHOLD: float  = float(os.getenv("MEMORY_IMPORTANCE_THRESHOLD", "0.4"))

# ── Topic Tracking ────────────────────────────────────────────────
TOPIC_SIMILARITY_THRESHOLD: float = float(os.getenv("TOPIC_SIMILARITY_THRESHOLD", "0.55"))

# ── Attention Keywords ────────────────────────────────────────────
_raw = os.getenv(
    "ATTENTION_KEYWORDS",
    "important,focus here,this concept,key point,remember,note that,crucial,essential,pay attention"
)
ATTENTION_KEYWORDS: list[str] = [k.strip().lower() for k in _raw.split(",")]

# ── Validation ────────────────────────────────────────────────────
def validate_config() -> None:
    if not OPENAI_API_KEY or not OPENAI_API_KEY.startswith("sk-"):
        raise EnvironmentError(
            "❌  OPENAI_API_KEY is missing or invalid.\n"
            "    Please copy .env.example → .env and add your API key."
        )