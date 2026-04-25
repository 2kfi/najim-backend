import yaml
import os
import glob


def has_stt_model(path: str) -> bool:
    return len(glob.glob(os.path.join(path, "*.bin"))) > 0


def has_tts_model(path: str) -> bool:
    return len(glob.glob(os.path.join(path, "*.onnx"))) > 0


def log(msg: str):
    print(f"[{msg}]")


with open("config.yaml") as f:
    config = yaml.safe_load(f)

# Models
MODELS_STORAGE_PATH = config["models"]["storage_path"]
MODELS_DOWNLOAD_ON_STARTUP = config["models"]["download_on_startup"]

# App
APP_HOST = config["app"]["host"]
APP_PORT = config["app"]["port"]
APP_LOG_LEVEL = config["app"]["log_level"]

# STT
STT_MODEL_PATH = config["stt"]["model_path"]
STT_HF_REPO = config["stt"]["hf_repo"]
STT_DEVICE = config["stt"]["device"]
STT_COMPUTE_TYPE = config["stt"]["compute_type"]
STT_BEAM_SIZE = config["stt"]["beam_size"]

# TTS - dynamically generated from config.yaml
TTS_CONFIGS = [
    {
        "lang": lang,
        "local_path": data["local_path"],
        "hf_repo": data["hf_repo"],
        "voice": data["voice"],
        "available": has_tts_model(data["local_path"]),
    }
    for lang, data in config["tts"].items()
]

# Settings
SETTINGS_VOLUME = config["settings"]["volume"]
SETTINGS_LENGTH_SCALE = config["settings"]["length_scale"]
SETTINGS_NOISE_SCALE = config["settings"]["noise_scale"]
SETTINGS_NOISE_W_SCALE = config["settings"]["noise_w_scale"]
SETTINGS_NORMALIZE_AUDIO = config["settings"]["normalize_audio"]
SETTINGS_NCHANNELS = config["settings"]["nchannels"]
SETTINGS_SAMPWIDTH = config["settings"]["sampwidth"]
SETTINGS_FRAMERATE = config["settings"]["framerate"]

# LLM
LLM_API_URL = config["llm"]["api_url"]
LLM_API_KEY = config["llm"]["api_key"]
LLM_MODEL = config["llm"]["model"]
LLM_TIMEOUT = config["llm"]["timeout"]

# MCP - dynamically generated from config.yaml
MCP_SERVERS = [
    {"url": url, "name": url.split("/")[-2]} for url in config["mcp"]["servers"]
]
MCP_MAX_RETRIES = config["mcp"]["max_retries"]


# ============ Model Checking & Downloading

STT_AVAILABLE = has_stt_model(STT_MODEL_PATH)


def check_and_download_models():
    global STT_AVAILABLE

    if not MODELS_DOWNLOAD_ON_STARTUP:
        return

    if not STT_AVAILABLE:
        log("DL STT missing, downloading...")
        if _download_model(STT_MODEL_PATH, STT_HF_REPO):
            STT_AVAILABLE = has_stt_model(STT_MODEL_PATH)

    for tts in TTS_CONFIGS:
        if not tts["available"]:
            log(f"DL {tts['lang']} TTS missing, downloading...")
            if _download_model(tts["local_path"], tts["hf_repo"]):
                tts["available"] = has_tts_model(tts["local_path"])


def _download_model(local_path: str, hf_repo: str) -> bool:
    try:
        from huggingface_hub import snapshot_download

        os.makedirs(local_path, exist_ok=True)
        snapshot_download(
            repo_id=hf_repo, local_dir=local_path, local_dir_use_symlinks=False
        )
        return True
    except Exception as e:
        log(f"DL-FAIL {hf_repo}: {e}")
        return False


def warn_missing_models():
    if not STT_AVAILABLE:
        log(f"MISSING STT: {STT_MODEL_PATH}")

    for tts in TTS_CONFIGS:
        if not tts["available"]:
            log(f"MISSING TTS ({tts['lang']}): {tts['local_path']}")


def validate_minimum():
    tts_available = sum(1 for tts in TTS_CONFIGS if tts["available"])

    if not STT_AVAILABLE:
        log("FATAL STT required but missing")
        return False

    if tts_available == 0:
        log("FATAL At least 1 TTS required but none available")
        return False

    return True


check_and_download_models()
warn_missing_models()

if not validate_minimum():
    log("FATAL Minimum models not available - exiting")
    exit(1)

log("READY App running - minimum models available")
