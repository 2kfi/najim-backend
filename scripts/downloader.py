#!/usr/bin/env python3
"""
Model downloader - downloads only needed model files from HuggingFace.
Fixed to use allow_patterns to avoid downloading entire 18GB repos.
"""
import os
import glob
import yaml
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = None) -> dict:
    if config_path is None:
        config_path = os.environ.get("CONFIG_PATH", "config.yaml")
    
    with open(config_path) as f:
        return yaml.safe_load(f)


def has_stt_model(path: str) -> bool:
    return len(glob.glob(os.path.join(path, "*.bin"))) > 0


def has_tts_model(path: str) -> bool:
    return len(glob.glob(os.path.join(path, "*.onnx"))) > 0


def voice_to_filenames(voice: str) -> dict:
    """
    Convert voice config to exact file names.
    
    voice: "en.en_GB.cori.high"  →  "ar.ar_JO.kareem.medium"
    returns:
        {
            "onnx": "en/en_GB/cori/high/en_GB-cori-high.onnx",
            "config": "en/en_GB/cori/high/en_GB-cori-high.onnx.json",
            "model_card": "en/en_GB/cori/high/MODEL_CARD"
        }
    """
    if not voice:
        return {"onnx": "*.onnx", "config": "*.onnx.json", "model_card": "MODEL_CARD"}
    
    parts = voice.split(".")
    if len(parts) < 4:
        logger.warning(f"Invalid voice format: {voice} (expected: lang.region.name.quality)")
        return {"onnx": "*.onnx", "config": "*.onnx.json", "model_card": "MODEL_CARD"}
    
    lang = parts[0]                        # "en"
    region = parts[1]                     # "en_GB"
    voice_name = parts[2]                  # "cori"
    quality = parts[3]                     # "high"
    
    repo_path = f"{lang}/{region}/{voice_name}/{quality}"
    filename = f"{region}-{voice_name}-{quality}"
    
    return {
        "onnx": f"{repo_path}/{filename}.onnx",
        "config": f"{repo_path}/{filename}.onnx.json",
        "model_card": f"{repo_path}/MODEL_CARD"
    }


def validate_tts_model(local_path: str, voice: str) -> bool:
    """Check all 3 required TTS files exist."""
    files = voice_to_filenames(voice)
    
    for key, pattern in [("ONNX", files["onnx"]), ("config", files["config"]), ("MODEL_CARD", files["model_card"])]:
        if key == "MODEL_CARD":
            model_card_path = os.path.join(local_path, files["model_card"])
            if not os.path.exists(model_card_path):
                logger.error(f"Missing MODEL_CARD: {model_card_path}")
                return False
        else:
            pattern_path = os.path.join(local_path, pattern)
            if not glob.glob(pattern_path):
                logger.error(f"Missing {key}: {pattern_path}")
                return False
    
    return True


def get_allow_patterns(model_type: str, hf_repo: str, voice: str = None) -> list:
    """Generate allow patterns to download only needed files."""
    
    if model_type == "stt":
        # Download all files for STT (selective download has issues with faster-whisper)
        return ["*"]
    
    elif model_type == "tts" and voice:
        # TTS - download only specific voice files
        # voice: "en.en_GB.cori.high" → exact filenames: "en_GB-cori-high.onnx"
        files = voice_to_filenames(voice)
        return [
            files["onnx"],
            files["config"],
            files["model_card"],
        ]
    
    return ["*"]


def get_ignore_patterns(model_type: str) -> list:
    """Patterns to exclude."""
    
    ignore = [
        ".git/*",
        "*.md",
        "README*",
        "*.txt",
        "*.pth",  # PyTorch weights we don't need
        "*.pt",
        "*.h5",   # Keras weights
    ]
    
    if model_type == "tts":
        ignore.extend([
            "original/*",      # Originalformat files
            "*/original/*",
            "*_original.onnx",
        ])
    
    return ignore


def download_model(local_path: str, hf_repo: str, model_type: str = "stt", voice: str = None) -> bool:
    """
    Download only specific model files using allow_patterns.
    
    Args:
        local_path: Local directory to save model
        hf_repo: HuggingFace repo ID (e.g., "Systran/faster-whisper-medium")
        model_type: "stt" or "tts"
        voice: TTS voice name (e.g., "en.en_GB.cori.high")
    
    Returns:
        True if download succeeded, False otherwise
    """
    try:
        from huggingface_hub import snapshot_download
        
        os.makedirs(local_path, exist_ok=True)
        
        allow_patterns = get_allow_patterns(model_type, hf_repo, voice)
        ignore_patterns = get_ignore_patterns(model_type)
        
        logger.info(f"Downloading {model_type}: {hf_repo}")
        logger.info(f"  Allow: {allow_patterns}")
        logger.info(f"  Ignore: {ignore_patterns}")
        logger.info(f"  To: {local_path}")
        
        snapshot_download(
            repo_id=hf_repo,
            local_dir=local_path,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            local_dir_use_symlinks=False,
        )
        
        # Validate TTS download
        if model_type == "tts" and voice:
            if not validate_tts_model(local_path, voice):
                logger.error(f"Validation failed - missing files: {local_path}")
                return False
        
        logger.info(f"✓ Download complete: {local_path}")
        return True
    
    except Exception as e:
        logger.error(f"✗ Download failed: {hf_repo} -> {e}")
        return False


def check_and_download_models(config: dict):
    """Check existing models and download missing ones."""
    
    # STT
    stt = config.get("stt", {})
    stt_path = stt.get("model_path", "models/whisper-medium")
    stt_repo = stt.get("hf_repo", "")
    
    download_on_startup = config.get("models", {}).get("download_on_startup", False)
    
    if not download_on_startup:
        logger.info("Auto-download disabled (download_on_startup: false)")
        logger.info("Use --force to download manually")
        return
    
    if not has_stt_model(stt_path):
        logger.info(f"STT model missing: {stt_path}")
        logger.info(f"  Repo: {stt_repo}")
        
        if download_model(stt_path, stt_repo, model_type="stt"):
            logger.info(f"✓ STT downloaded: {stt_path}")
        else:
            logger.error(f"✗ STT download failed")
    else:
        logger.info(f"✓ STT exists: {stt_path}")
    
    # TTS
    tts_configs = config.get("tts", {})
    
    for lang, data in tts_configs.items():
        local_path = data.get("local_path", "")
        hf_repo = data.get("hf_repo", "")
        voice = data.get("voice", "")
        
        if not local_path or not hf_repo or not voice:
            continue
        
        if has_tts_model(local_path):
            logger.info(f"✓ TTS [{lang}] exists: {local_path}")
            continue
        
        logger.info(f"TTS [{lang}] missing: {local_path}")
        logger.info(f"  Repo: {hf_repo}")
        logger.info(f"  Voice: {voice}")
        
        if download_model(local_path, hf_repo, model_type="tts", voice=voice):
            logger.info(f"✓ TTS [{lang}] downloaded: {local_path}")
        else:
            logger.error(f"✗ TTS [{lang}] download failed")


def warn_missing_models(config: dict):
    """Warn about missing models without downloading."""
    
    stt = config.get("stt", {})
    stt_path = stt.get("model_path", "models/whisper-medium")
    
    if not has_stt_model(stt_path):
        logger.warning(f"MISSING STT: {stt_path}")
    
    tts_configs = config.get("tts", {})
    
    for lang, data in tts_configs.items():
        local_path = data.get("local_path", "")
        voice = data.get("voice", "")
        if not local_path:
            continue
        
        has_onnx = has_tts_model(local_path)
        has_all = validate_tts_model(local_path, voice) if voice else has_onnx
        
        if not has_all:
            logger.warning(f"MISSING TTS [{lang}]: {local_path}")
            if not has_onnx:
                logger.warning(f"  -> Missing .onnx file")
            if voice:
                model_card = os.path.join(local_path, "MODEL_CARD")
                if not os.path.exists(model_card):
                    logger.warning(f"  -> Missing MODEL_CARD")


def validate_minimum(config: dict) -> bool:
    """Validate minimum required models exist."""
    
    stt = config.get("stt", {})
    stt_path = stt.get("model_path", "models/whisper-medium")
    
    tts_configs = config.get("tts", {})
    
    stt_ok = has_stt_model(stt_path)
    
    tts_ok = False
    for lang, data in tts_configs.items():
        local_path = data.get("local_path", "")
        voice = data.get("voice", "")
        if not local_path:
            continue
        
        if validate_tts_model(local_path, voice):
            tts_ok = True
            break
    
    if not stt_ok:
        logger.error("FATAL: STT model missing but required")
        return False
    
    if not tts_ok:
        logger.error("FATAL: No valid TTS voices available")
        return False
    
    return True
    
    if not tts_ok:
        logger.error("FATAL: No TTS voices available")
        return False
    
    return True


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Download AI models from HuggingFace")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--check", action="store_true", help="Check existing models only")
    parser.add_argument("--force", action="store_true", help="Force re-download")
    parser.add_argument("--quiet", action="store_true", help="Suppress info output")
    args = parser.parse_args()
    
    if args.quiet:
        logging.basicConfig(level=logging.ERROR)
    
    config = load_config(args.config)
    
    if args.check:
        warn_missing_models(config)
        validate_minimum(config)
        return
    
    # Check if models exist
    stt = config.get("stt", {})
    stt_path = stt.get("model_path", "")
    
    if not args.force and has_stt_model(stt_path):
        logger.info(f"STT already exists: {stt_path}")
        logger.info("Use --force to re-download")
        
        tts_configs = config.get("tts", {})
        for lang, data in tts_configs.items():
            local_path = data.get("local_path", "")
            if local_path and has_tts_model(local_path):
                logger.info(f"TTS [{lang}] already exists: {local_path}")
        
        return
    
    # Download models
    logger.info("=" * 50)
    logger.info("Starting model download")
    logger.info("=" * 50)
    
    check_and_download_models(config)
    
    # Validate
    logger.info("=" * 50)
    logger.info("Validating models")
    logger.info("=" * 50)
    
    if not validate_minimum(config):
        logger.error("FAILED: Minimum models not available")
        exit(1)
    
    logger.info("SUCCESS: All required models downloaded")


if __name__ == "__main__":
    main()