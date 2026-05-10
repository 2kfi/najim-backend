# Najim

## What is Najim?

Open-source voice home assistant API.

## What's Najim Eco-system?

- integrates with Home Assistant (soon)
- Has a client for file search and simple interactions with systems (soon)
- a voice assistant client to replace Alexa or Google home
- a smarter AI responses

## Why should i use Najim?

- modeler, so you can replace every component if you want
- open-source
- 100% local
- privacy focused
- runs on low-end devices
- has a client for every Operation system (soon to be done)

## How to install?

```bash
# Clone the repository
git clone https://github.com/2kfi/najim-backend
cd najim-backend

# Install dependencies
pip install -r requirements.txt

# Copy config and env files (make sure to copy config and .env for docker)
cp config.example.yaml config.yaml
cp .env.example .env

# Edit config.yaml with your settings (see Configuration section)
# Edit .env with your environment variables

# Run the server
python app.py
```

## Configuration

Edit `config.yaml` with your settings:

```yaml
models:
  storage_path: "./models"
  download_on_startup: true

api:
  host: "0.0.0.0"           # server bind address
  port: 8080               # server port
  base_path: "/api/v1"     # API prefix
  max_audio_size_mb: 10    # max upload size
  allowed_audio_types:    # allowed audio formats
    - "audio/wav"
    - "audio/mpeg"
    - "audio/ogg"

auth:
  default_rate_limit: 60   # default requests per minute
  api_keys:                # your API keys
    "sk-dev-001":
      name: dev-client
      rate_limit: 100

stt:
  model_path: "models/whisper-medium"
  hf_repo: "Systran/faster-whisper-medium"
  device: "cpu"            # cpu, cuda, auto
  compute_type: "int8"     # int8, fp16, fp32
  task: "transcribe"       # transcribe or translate
  beam_size: 5
  vad_filter: true
  vad_threshold: 0.5
  vad_min_speech_duration_ms: 250
  vad_min_silence_duration_ms: 200

settings:
  volume: 0.75          # Increased for clarity without digital clipping
  length_scale: 1.0     # Keep at 1.0; adjust to 1.1 if the voice feels too rushed
  noise_scale: 0.75     # Slightly higher than default to reduce robotic "dryness"
  noise_w_scale: 0.5    # Reduced to 0.5 to stop the voice from sounding "wobbly"
  normalize_audio: true # Keeps volume consistent across different sentences
  nchannels: 1
  sampwidth: 2
  framerate: 22050      # Most high-quality Piper models are native to 22050Hz
  use_cuda: false       # Keep false unless you have an NVIDIA GPU


tts:
  en:
    local_path: "models/TTS-CORI-EN"
    use_cuda: false
  ar:
    local_path: "./models/TTS-KAREEM-ARABIC"
    use_cuda: false

llm:
  api_url: ""              # e.g. https://api.groq.com/openai/v1
  api_key: ""              # your API key
  model: "gpt-4o-mini"    # model name
  timeout: 120

mcp:
  servers:                 # MCP server URLs (optional)
    - url: "http://localhost:1241/sse"
  sse_read_timeout: 300.0
  connect_timeout: 30.0
  tool_timeout: 30.0
  max_retries: 2
  max_tool_loops: 5

observability:
  log_level: "INFO"        # DEBUG, INFO, WARNING, ERROR
  log_format: "json"       # json or text
  metrics_enabled: true   # enable /metrics endpoint
  tracing_enabled: false
```

## Testing

```bash
# Install test dependencies (already in requirements.txt)
pip install pytest pytest-asyncio

# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_app.py -v

# Run with verbose output
pytest tests/ -vv
```

## API

```
- POST /api/v1/process - process audio
- GET /api/v1/health - health check  
- GET /api/v1/metrics - metrics
```

## To Do list