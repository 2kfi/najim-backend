"""
Unit tests for Najim Backend API
"""
import pytest
from unittest.mock import Mock, patch, AsyncMock
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestConfigLoading:
    """Test configuration loading"""
    
    def test_load_config(self):
        """Test that config loads correctly"""
        import yaml
        config_path = os.environ.get("CONFIG_PATH", "config.yaml")
        assert os.path.exists(config_path), f"Config file not found: {config_path}"
        
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        assert config is not None
        assert "api" in config
        assert "stt" in config
        assert "tts" in config
        assert "llm" in config


class TestSettings:
    """Test Settings class"""
    
    def test_settings_defaults(self):
        """Test default settings values"""
        from app import Settings, config
        
        settings = Settings()
        
        assert settings.API_HOST == "0.0.0.0"
        assert settings.API_PORT == 8080
        assert settings.BASE_PATH == "/api/v1"
        assert settings.MAX_AUDIO_SIZE_MB == 10
        assert settings.DEFAULT_RATE_LIMIT == 60


class TestHelpers:
    """Test helper functions"""
    
    def test_find_onnx_file(self):
        """Test ONNX file finder"""
        from app import find_onnx_file
        
        with patch("glob.glob") as mock_glob:
            mock_glob.return_value = ["/path/to/model.onnx"]
            result = find_onnx_file("/some/path")
            assert result == "/path/to/model.onnx"
    
    def test_find_onnx_file_not_found(self):
        """Test ONNX file finder raises when not found"""
        from app import find_onnx_file
        
        with patch("glob.glob") as mock_glob:
            mock_glob.return_value = []
            with pytest.raises(FileNotFoundError):
                find_onnx_file("/empty/path")


class TestAppState:
    """Test AppState class"""
    
    def test_app_state_init(self):
        """Test AppState initializes correctly"""
        from app import AppState
        
        state = AppState()
        
        assert state.whisper_model is None
        assert state.tts_voices == {}
        assert state.tts_voice_paths == {}
        assert state.llm_client is None
        assert state.mcp_wrapper is None
        assert not state.initialized
        assert state._llm_cache == {}


class TestPydanticModels:
    """Test Pydantic models"""
    
    def test_health_response_model(self):
        """Test HealthResponse model"""
        from app import HealthResponse
        
        response = HealthResponse(
            status="healthy",
            stt_model="models/whisper-medium",
            stt_device="cpu",
            tts_voices=["en", "ar"],
            mcp_servers=["http://localhost:1241"],
            llm_url="https://api.groq.com/openai/v1"
        )
        
        assert response.status == "healthy"
        assert response.stt_model == "models/whisper-medium"
        assert len(response.tts_voices) == 2
    
    def test_error_response_model(self):
        """Test ErrorResponse model"""
        from app import ErrorResponse
        
        response = ErrorResponse(
            error="Something went wrong",
            error_code="INTERNAL_ERROR",
            request_id="test-123"
        )
        
        assert response.error == "Something went wrong"
        assert response.error_code == "INTERNAL_ERROR"
        assert response.request_id == "test-123"


class TestMCPSessionManager:
    """Test MCP session manager"""
    
    @pytest.mark.asyncio
    async def test_session_manager_init(self):
        """Test MCPSessionManager initialization"""
        from scripts.mcp import MCPSessionManager
        
        mgr = MCPSessionManager(
            url="http://localhost:8080/sse",
            api_key="test-key",
            sse_read_timeout=30.0,
            connect_timeout=10.0,
            tool_timeout=60.0
        )
        
        assert mgr.url == "http://localhost:8080/sse"
        assert mgr.api_key == "test-key"
        assert mgr.sse_read_timeout == 30.0
        assert mgr.connect_timeout == 10.0
        assert mgr.tool_timeout == 60.0
        assert not mgr.connected


class TestMCPWrapper:
    """Test MCP wrapper"""
    
    def test_mcp_wrapper_init(self):
        """Test MCPWrapper initialization"""
        from scripts.mcp import MCPWrapper
        
        wrapper = MCPWrapper(
            llama_base_url="http://localhost:8080/v1",
            llama_model="llama-3.3-70b-versatile",
            mcp_servers=[],
            api_key="test-key",
            timeout=60.0,
            max_tool_loops=5,
            max_retries=2
        )
        
        assert wrapper.llama_model == "llama-3.3-70b-versatile"
        assert wrapper.max_tool_loops == 5
        assert wrapper.max_retries == 2
        assert wrapper.mcp_managers == []
    
    def test_mcp_wrapper_with_servers(self):
        """Test MCPWrapper with servers config"""
        from scripts.mcp import MCPWrapper
        
        servers = [
            {"url": "http://localhost:1241/sse", "api_key": "key1"},
            {"url": "http://localhost:1243/sse", "api_key": "key2"}
        ]
        
        wrapper = MCPWrapper(
            llama_base_url="http://localhost:8080/v1",
            llama_model="llama-3.3-70b-versatile",
            mcp_servers=servers,
            api_key="test-key",
            timeout=60.0,
            max_tool_loops=5,
            max_retries=2
        )
        
        assert len(wrapper.mcp_managers) == 2
        assert wrapper.mcp_managers[0].url == "http://localhost:1241/sse"
        assert wrapper.mcp_managers[1].url == "http://localhost:1243/sse"


class TestRateLimiting:
    """Test rate limiting functionality"""
    
    def test_limiter_init(self):
        """Test limiter is initialized"""
        from app import limiter
        
        assert limiter is not None


class TestLogging:
    """Test logging configuration"""
    
    def test_logger_init(self):
        """Test logger is initialized"""
        from app import logger, settings
        
        assert logger is not None
        assert logger.level == getattr(logging, settings.LOG_LEVEL)


class TestFileHelpers:
    """Test file helper functions"""
    
    def test_combine_wav_files_empty(self):
        """Test combine_wav_files raises on empty input"""
        from app import combine_wav_files
        
        with pytest.raises(ValueError):
            combine_wav_files([], "/tmp/output.wav")


@pytest.mark.asyncio
class TestAsyncFunctions:
    """Test async functions"""
    
    async def test_cleanup_files(self):
        """Test cleanup handles missing files"""
        from app import cleanup_files
        
        result = await cleanup_files("/nonexistent/file.wav")
        assert result is None
    
    async def test_synthesize_one_no_voice(self):
        """Test synthesize_one raises without voice"""
        from app import synthesize_one, state
        
        with pytest.raises(ValueError):
            await synthesize_one("test", None, "/tmp/output.wav")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])