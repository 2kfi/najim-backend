import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status

from core.jwt_auth import verify_jwt
from core.redis_manager import get_redis
from core.schemas import CreateSessionRequest, CreateSessionResponse, SessionData, SessionConfig
from sessions.session_registry import SessionRegistry
from sessions.conversation_store import ConversationStore
from sessions.device_registry import DeviceRegistry
from sessions.permissions import PermissionStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["sessions"])


@router.post("/sessions", response_model=CreateSessionResponse, status_code=status.HTTP_201_CREATED)
async def create_session(
    request: CreateSessionRequest,
    claims: dict = Depends(verify_jwt),
):
    if claims["device_id"] != request.device_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Device ID mismatch")

    redis = await get_redis()
    session_reg = SessionRegistry(redis)

    existing = await session_reg.get(request.device_id)
    if existing:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Session already exists")

    session = await session_reg.create(
        device_id=request.device_id,
        user_id=request.user_id or claims["user_id"],
        config=request.config,
    )
    logger.info(f"Session created for device {request.device_id}")

    return CreateSessionResponse(
        session_id=request.device_id,
        device_id=request.device_id,
        created_at=session.created_at,
        message="Session created successfully",
    )


@router.get("/sessions/{device_id}", response_model=SessionData)
async def get_session(device_id: str, claims: dict = Depends(verify_jwt)):
    if claims["device_id"] != device_id and claims["user_id"] != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")

    redis = await get_redis()
    session_reg = SessionRegistry(redis)
    session = await session_reg.get(device_id)

    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    return session


@router.delete("/sessions/{device_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(device_id: str, claims: dict = Depends(verify_jwt)):
    if claims["device_id"] != device_id and claims["user_id"] != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")

    redis = await get_redis()
    session_reg = SessionRegistry(redis)
    conv_store = ConversationStore(redis)

    await conv_store.clear(device_id)
    await session_reg.delete(device_id)
    logger.info(f"Session deleted for device {device_id}")
    return None


@router.get("/sessions", response_model=list[SessionData])
async def list_sessions(claims: dict = Depends(verify_jwt)):
    redis = await get_redis()
    session_reg = SessionRegistry(redis)
    return await session_reg.list_all()


@router.get("/conversations/{device_id}")
async def get_conversation(device_id: str, limit: int = 50, claims: dict = Depends(verify_jwt)):
    if claims["device_id"] != device_id and claims["user_id"] != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")

    redis = await get_redis()
    conv_store = ConversationStore(redis)
    history = await conv_store.get_history(device_id, limit)
    return {"device_id": device_id, "messages": [m.model_dump() for m in history], "count": len(history)}


@router.get("/devices", response_model=list[dict])
async def list_devices(claims: dict = Depends(verify_jwt)):
    redis = await get_redis()
    device_reg = DeviceRegistry(redis)
    devices = await device_reg.list_connected()
    return [d.model_dump() for d in devices]


@router.get("/devices/{device_id}")
async def get_device(device_id: str, claims: dict = Depends(verify_jwt)):
    redis = await get_redis()
    device_reg = DeviceRegistry(redis)
    info = await device_reg.get(device_id)
    if not info:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Device not found")
    return info.model_dump()


@router.patch("/sessions/{device_id}/config")
async def update_session_config(device_id: str, config: SessionConfig, claims: dict = Depends(verify_jwt)):
    if claims["device_id"] != device_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")

    redis = await get_redis()
    session_reg = SessionRegistry(redis)
    await session_reg.update_config(device_id, config)
    return {"status": "updated", "device_id": device_id}


@router.get("/permissions/{device_id}")
async def get_permissions(device_id: str, claims: dict = Depends(verify_jwt)):
    redis = await get_redis()
    perms = PermissionStore(redis)
    return await perms.get_all(device_id)


@router.put("/permissions/{device_id}/{tool_name}")
async def set_permission(device_id: str, tool_name: str, allowed: bool, claims: dict = Depends(verify_jwt)):
    if claims["user_id"] != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin only")

    redis = await get_redis()
    perms = PermissionStore(redis)
    await perms.set_permission(device_id, tool_name, allowed)
    return {"device_id": device_id, "tool_name": tool_name, "allowed": allowed}