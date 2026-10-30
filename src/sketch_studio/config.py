import os
from pathlib import Path

ROOT = Path(os.environ.get("STUDIO_HOME", Path(__file__).resolve().parents[2]))
INBOX = ROOT / "inbox"
STORAGE = ROOT / "storage"
DB_PATH = STORAGE / "studio_local.db"

WEB_HOST = "127.0.0.1"
WEB_PORT = int(os.environ.get("STUDIO_WEB_PORT", "8765"))
WEB_URL = f"http://{WEB_HOST}:{WEB_PORT}"

# Default COMFY_URL; the live value is stored in SQLite via set_comfy_url (Colab tunnel URL changes every session).
COMFY_URL = os.environ.get("COMFY_URL", "")
# Self-hosted VLM (Ollama on Colab) for the web-only flow; empty model = first model Ollama reports.
LLM_URL = os.environ.get("LLM_URL", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "")

# Model files the Colab notebook downloads (SD1.5 family: fast on a T4).
CHECKPOINT = os.environ.get("STUDIO_CHECKPOINT", "DreamShaper_8_pruned.safetensors")
CONTROLNET_LINEART = "control_v11p_sd15_lineart_fp16.safetensors"
CONTROLNET_SCRIBBLE = "control_v11p_sd15_scribble_fp16.safetensors"

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".heic", ".heif", ".bmp"}

for d in (INBOX, STORAGE):
    d.mkdir(parents=True, exist_ok=True)
