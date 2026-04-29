#!/usr/bin/env python3
"""
Model downloader - downloads only needed model files from HuggingFace.
Fixed to extract flat files and destroy the nested directory tree!
"""
import os
import glob
import yaml
import logging
import shutil

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
    Convert voice config to exact file names, keeping track of both
    the nested repo path (for downloading) and the flat path (for moving/validating).
    """
    if not voice:
        return {
            "onnx_repo": "*.onnx", "config_repo": "*.onnx.json", "model_card_repo": "MODEL_CARD",
            "onnx_flat": "*.onnx", "config_flat": "*.onnx.json", "model_card_flat": "MODEL_CARD"
        }
    
    parts = voice.split(".")
    if len(parts) < 4:
        logger.warning(f"Invalid voice format: {voice} (expected: lang.region.name.quality)")
        return {
            "onnx_repo": "*.onnx", "config_repo": "*.onnx.json", "model_card_repo": "MODEL_CARD",
            "onnx_flat": "*.onnx", "config_flat": "*.onnx.json", "model_card_flat": "MODEL_CARD"
        }
    
    lang = parts[0]
    region = parts[1]
    voice_name = parts[2]
    quality = parts[3]
    
    repo_path = f"{lang}/{region}/{voice_name}/{quality}"
    filename = f"{region}-{voice_name}-{quality}"
    
    return {
        "onnx_repo": f"{repo_path}/{filename}.onnx",
        "config_repo": f"{repo_path}/{filename}.onnx.json",
        "model_card_repo": f"{repo_path}/MODEL_CARD",
        "onnx_flat": f"{filename}.onnx",
        "config_flat": f"{filename}.onnx.json",
        "model_card_flat": "MODEL_CARD"
    }


def validate_tts_model(local_path: str, voice: str) -> bool:
    """Check all 3 required TTS files exist flatly in the directory."""
    files = voice_to_filenames(voice)
    
    for key, expected_file in [("ONNX", files["onnx_flat"]), ("config", files["config_flat"]), ("MODEL_CARD", files["model_card_flat"])]:
        file_path = os.path.join(local_path, expected_file)
        
        if '*' in expected_file:
            if not glob.glob(file_path):
                logger.error(f"Missing {key}: {file_path}")
                return False
        else:
            if not os.path.exists(file_path):
                logger.error(f"Missing {key}: {file_path}")
                return False
    
    return True


def get_allow_patterns(model_type: str, hf_repo: str, voice: str = None) -> list:
    """Generate allow patterns to download only needed files."""
    if model_type == "stt":
        return ["*"]
    
    elif model_type == "tts" and voice:
        files = voice_to_filenames(voice)
        return [
            files["onnx_repo"],
            files["config_repo"],
            files["model_card_repo"],
        ]
    
    return ["*"]


def get_ignore_patterns(model_type: str) -> list:
    """Patterns to exclude."""
    
    if model_type == "stt":
        return []
    
    ignore = [
        ".git/*",
        "*.md",
        "README*",
        "*.txt",
        "*.pth",
        "*.pt",
        "*.h5",
    ]
    
    if model_type == "tts":
        ignore.extend([
            "original/*",
            "*/original/*",
            "*_original.onnx",
        ])
    
    return ignore


def download_model(local_path: str, hf_repo: str, model_type: str = "stt", voice: str = None) -> bool:
    try:
        from huggingface_hub import snapshot_download
        
        os.makedirs(local_path, exist_ok=True)
        
        allow_patterns = get_allow_patterns(model_type, hf_repo, voice)
        ignore_patterns = get_ignore_patterns(model_type)
        
        logger.info(f"Downloading {model_type}: {hf_repo}")
        
        snapshot_download(
            repo_id=hf_repo,
            local_dir=local_path,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            local_dir_use_symlinks=False,
        )
        
        # FLATTEN DIRECTORY TREE FOR TTS
        if model_type == "tts" and voice:
            files = voice_to_filenames(voice)
            
            # 1. Move files out of the nested tree to the root of local_path
            for key in ["onnx", "config", "model_card"]:
                src = os.path.join(local_path, files[f"{key}_repo"])
                dst = os.path.join(local_path, files[f"{key}_flat"])
                
                if os.path.exists(src) and src != dst:
                    shutil.move(src, dst)
                    logger.info(f"Flattened: {files[f'{key}_flat']}")
            
            # 2. Obliterate the empty nested directory tree
            top_level_dir = files["onnx_repo"].split('/')[0]
            dir_to_remove = os.path.join(local_path, top_level_dir)
            
            if os.path.exists(dir_to_remove) and os.path.isdir(dir_to_remove):
                shutil.rmtree(dir_to_remove)
                logger.info(f"Cleaned up empty directory tree: {top_level_dir}/")

            if not validate_tts_model(local_path, voice):
                logger.error(f"Validation failed - missing flat files: {local_path}")
                return False
        
        logger.info(f"✓ Download complete: {local_path}")
        return True
    
    except Exception as e:
        logger.error(f"✗ Download failed: {hf_repo} -> {e}")
        return False


def check_and_download_models(config: dict):
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
        
        if validate_tts_model(local_path, voice):
            logger.info(f"✓ TTS [{lang}] exists (flat): {local_path}")
            continue
        
        logger.info(f"TTS [{lang}] missing or not flat: {local_path}")
        if download_model(local_path, hf_repo, model_type="tts", voice=voice):
            logger.info(f"✓ TTS [{lang}] downloaded and flattened: {local_path}")
        else:
            logger.error(f"✗ TTS [{lang}] download failed")


def warn_missing_models(config: dict):
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
            logger.warning(f"MISSING TTS [{lang}]: {local_path} (Are they flat?)")


def validate_minimum(config: dict) -> bool:
    stt = config.get("stt", {})
    stt_path = stt.get("model_path", "models/whisper-medium")
    tts_configs = config.get("tts", {})
    
    stt_ok = has_stt_model(stt_path)
    tts_ok = False
    
    for lang, data in tts_configs.items():
        local_path = data.get("local_path", "")
        voice = data.get("voice", "")
        if local_path and validate_tts_model(local_path, voice):
            tts_ok = True
            break
    
    if not stt_ok:
        logger.error("FATAL: STT model missing but required")
        return False
    
    if not tts_ok:
        logger.error("FATAL: No valid TTS voices available")
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
    
    if not args.force:
        # Check STT
        stt = config.get("stt", {})
        stt_path = stt.get("model_path", "")
        stt_ok = has_stt_model(stt_path)
        
        # Check TTS
        tts_configs = config.get("tts", {})
        tts_all_ok = True
        for lang, data in tts_configs.items():
            local_path = data.get("local_path", "")
            voice = data.get("voice", "")
            if local_path and not validate_tts_model(local_path, voice):
                tts_all_ok = False
                break
                
        if stt_ok and tts_all_ok:
            logger.info("All models already exist in flat format.")
            logger.info("Use --force to re-download")
            return
    
    logger.info("=" * 50)
    logger.info("Starting model download")
    logger.info("=" * 50)
    
    check_and_download_models(config)
    
    logger.info("=" * 50)
    logger.info("Validating models")
    logger.info("=" * 50)
    
    if not validate_minimum(config):
        logger.error("FAILED: Minimum models not available")
        exit(1)
    
    logger.info("SUCCESS: All required models downloaded")


if __name__ == "__main__":
    main()