from datetime import datetime
from typing import Optional
from core.redis_manager import RedisManager
from core.schemas import Message, MessageRole
from core.config import get_settings


class ConversationStore:
    KEY_PREFIX = "conv"

    def __init__(self, redis: RedisManager):
        self.redis = redis
        self._settings = get_settings()

    def _key(self, device_id: str) -> str:
        return f"{self.KEY_PREFIX}:{device_id}"

    async def add_message(self, device_id: str, role: MessageRole, content: str, tool_call_id: Optional[str] = None, tool_name: Optional[str] = None) -> int:
        msg = Message(role=role, content=content, tool_call_id=tool_call_id, tool_name=tool_name)
        msg_json = msg.model_dump_json()
        length = await self.redis.rpush(self._key(device_id), msg_json)
        current_len = await self.redis.client.llen(self._key(device_id))
        if current_len > self._settings.session.max_history:
            trim_start = max(0, current_len - self._settings.session.max_history)
            await self.redis.ltrim(self._key(device_id), trim_start, -1)
        return length

    async def get_history(self, device_id: str, limit: Optional[int] = None) -> list[Message]:
        max_len = limit or self._settings.session.max_history
        raw = await self.redis.lrange(self._key(device_id), -max_len, -1)
        messages = []
        for item in raw:
            if isinstance(item, dict):
                messages.append(Message.model_validate(item))
            elif isinstance(item, str):
                messages.append(Message.model_validate_json(item))
            else:
                messages.append(Message.model_validate_json(item.decode() if isinstance(item, bytes) else item))
        return messages

    async def get_history_for_llm(self, device_id: str, limit: int = 50) -> list[dict[str, str]]:
        messages = await self.get_history(device_id, limit)
        out = []
        for m in messages:
            msg_dict = {"role": m.role.value, "content": m.content}
            if m.tool_call_id:
                msg_dict["tool_call_id"] = m.tool_call_id
                msg_dict["name"] = m.tool_name or ""
            out.append(msg_dict)
        return out

    async def add_tool_result(self, device_id: str, tool_call_id: str, tool_name: str, result: str) -> int:
        return await self.add_message(device_id, MessageRole.TOOL, result, tool_call_id=tool_call_id, tool_name=tool_name)

    async def clear(self, device_id: str) -> None:
        await self.redis.delete(self._key(device_id))

    async def count(self, device_id: str) -> int:
        return await self.redis.client.llen(self._key(device_id))