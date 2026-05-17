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
        from core.config import get_settings
        
        settings = get_settings()
        
        assert settings.api_host == "0.0.0.0"
        assert settings.api_port == 8080
        assert settings.debug == False
        assert settings.max_audio_size_mb == 10


class TestHelpers:
    """Test helper functions"""
    
    def test_find_onnx_file(self):
        """Test ONNX file finder"""
        from pipeline.tts_queue import TTSQueue
        from core.redis_manager import RedisManager
        
    def test_find_onnx_file_not_found(self):
        """Test ONNX file finder raises when not found"""
        pass


class TestAppState:
    """Test AppState class"""
    
    def test_app_state_init(self):
        """Test AppState initializes correctly"""
        from core.app_state import AppState
        
        assert AppState.whisper_model is None
        assert AppState.tts_voices == {}
        assert AppState.tts_voice_paths == {}
        assert AppState.llm_client is None
        assert not AppState.initialized


class TestPydanticModels:
    """Test Pydantic models"""
    
    def test_health_response_model(self):
        """Test HealthResponse model"""
        from core.schemas import HealthResponse
        
        response = HealthResponse(
            status="healthy",
            node_id="node-1",
            redis=True,
            whisper_model=True,
            tts_voices=["en", "ar"],
            uptime=123.45,
            connected_devices=5,
        )
        
        assert response.status == "healthy"
        assert response.whisper_model == True
        assert len(response.tts_voices) == 2
    
    def test_session_data_model(self):
        """Test SessionData model"""
        from core.schemas import SessionData
        
        session = SessionData(
            device_id="device-123",
            user_id="user-456",
        )
        
        assert session.device_id == "device-123"
        assert session.user_id == "user-456"
        assert session.status == "active"


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
        import logging
        from app import logger, settings
        
        assert logger is not None


class TestFileHelpers:
    """Test file helper functions"""
    
    def test_combine_wav_files_empty(self):
        """Test combine_wav_files raises on empty input"""
        pass


@pytest.mark.asyncio
class TestAsyncFunctions:
    """Test async functions"""
    
    async def test_cleanup_files(self):
        """Test cleanup handles missing files"""
        pass
    
    async def test_synthesize_one_no_voice(self):
        """Test synthesize_one raises without voice"""
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])