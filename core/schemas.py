import base64
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import UUID, uuid4
from pydantic import BaseModel, Field


class WSMessageType(str, Enum):
    CONNECT = "connect"
    DISCONNECT = "disconnect"
    TOOL_REQUEST = "tool_request"
    TOOL_RESPONSE = "tool_response"
    AUDIO = "audio"
    AUDIO_CHUNK = "audio_chunk"
    TRANSCRIPT = "transcript"
    HEARTBEAT = "heartbeat"
    ERROR = "error"


class ToolResultStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"


class DeviceStatus(str, Enum):
    ONLINE = "online"
    OFFLINE = "offline"
    BUSY = "busy"


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class SessionConfig(BaseModel):
    language: Optional[str] = None
    tts_voice: str = "en"
    stt_task: Optional[str] = None
    tool_permissions: dict[str, str] = Field(default_factory=dict)


class Message(BaseModel):
    role: MessageRole
    content: str
    tool_call_id: Optional[str] = None
    tool_name: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class SessionData(BaseModel):
    device_id: str
    user_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_active: datetime = Field(default_factory=datetime.utcnow)
    config: SessionConfig = Field(default_factory=SessionConfig)
    status: str = "active"
    language: Optional[str] = None


class ToolRequest(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    tool_name: str
    params: dict[str, Any] = Field(default_factory=dict)
    correlation_id: str = Field(default_factory=uuid4)


class ToolResponse(BaseModel):
    correlation_id: str
    status: ToolResultStatus
    result: Optional[Any] = None
    error: Optional[str] = None


class WSMessage(BaseModel):
    type: WSMessageType
    device_id: Optional[str] = None
    correlation_id: Optional[str] = None
    payload: Optional[dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class AudioJob(BaseModel):
    device_id: str
    session_id: str
    audio_data: str  # base64 encoded
    chunk_index: int = 0
    total_chunks: int = 1
    sample_rate: int = 16000
    channels: int = 1


class DeviceInfo(BaseModel):
    device_id: str
    user_id: Optional[str] = None
    host: Optional[str] = None
    port: Optional[int] = None
    capabilities: list[str] = Field(default_factory=list)
    status: DeviceStatus = DeviceStatus.ONLINE
    connected_at: datetime = Field(default_factory=datetime.utcnow)
    last_heartbeat: datetime = Field(default_factory=datetime.utcnow)
    node_id: Optional[str] = None


class ToolCallRecord(BaseModel):
    correlation_id: str
    tool_name: str
    params: dict[str, Any]
    device_id: str
    session_id: str
    initiated_at: datetime = Field(default_factory=datetime.utcnow)
    status: str = "pending"
    result: Optional[Any] = None
    error: Optional[str] = None
    completed_at: Optional[datetime] = None


class PipelineEvent(BaseModel):
    event_type: str
    device_id: str
    session_id: str
    data: dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    source_node: Optional[str] = None


class CreateSessionRequest(BaseModel):
    device_id: str
    user_id: str
    config: Optional[SessionConfig] = None
    capabilities: list[str] = Field(default_factory=list)


class CreateSessionResponse(BaseModel):
    session_id: str
    device_id: str
    created_at: datetime
    message: str = "Session created successfully"


class HealthResponse(BaseModel):
    status: str
    node_id: str
    redis: bool
    whisper_model: bool
    tts_voices: list[str]
    uptime: float
    connected_devices: int


class ToolDefinition(BaseModel):
    name: str
    description: str
    input_schema: dict[str, Any]
    is_internal: bool = True
    device_capability: Optional[str] = None


class ToolCallResult(BaseModel):
    tool_name: str
    success: bool
    result: Optional[Any] = None
    error: Optional[str] = None
    duration_ms: int = 0