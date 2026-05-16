import asyncio
import json
import logging
from typing import Any, Optional

from core.app_state import get_app_state
from core.config import get_settings
from core.redis_manager import RedisManager
from core.schemas import MessageRole, ToolCallResult
from sessions.conversation_store import ConversationStore
from tools.router import route_tool_calls_batch, UnknownToolError

logger = logging.getLogger(__name__)


class LLMRunner:
    def __init__(self, redis: RedisManager, conversation_store: ConversationStore):
        self.redis = redis
        self.conversation_store = conversation_store
        self._settings = get_settings()
        self._max_tool_loops = self._settings.mcp.max_tool_loops

    async def run_query(
        self,
        device_id: str,
        user_message: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        client = get_app_state().get_llm_client()
        if not client:
            raise RuntimeError("LLM client not initialized")

        history = await self.conversation_store.get_history_for_llm(device_id)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_message})
        for h in history:
            messages.append(h)

        tools_schema = await self._get_tools_schema()

        for iteration in range(self._max_tool_loops):
            response = await client.chat.completions.create(
                model=self._settings.llm.model,
                messages=messages,
                tools=tools_schema if tools_schema else None,
                tool_choice="auto",
            )
            message = response.choices[0].message

            if not message.tool_calls:
                if iteration == 0:
                    await self.conversation_store.add_message(device_id, MessageRole.USER, user_message)
                await self.conversation_store.add_message(device_id, MessageRole.ASSISTANT, message.content or "")
                return message.content or ""

            if iteration == 0:
                await self.conversation_store.add_message(device_id, MessageRole.USER, user_message)
            messages.append(message.model_dump(exclude_none=True))

            tool_results = await route_tool_calls_batch(device_id, device_id, [
                tc.model_dump() for tc in message.tool_calls
            ])

            for i, tc_result in enumerate(tool_results):
                tc = message.tool_calls[i]
                tool_name = tc.function.name
                content = json.dumps(tc_result.result) if tc_result.success else f"Error: {tc_result.error}"
                await self.conversation_store.add_tool_result(device_id, tc.id, tool_name, content)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": tool_name,
                    "content": content,
                })

        final_response = await client.chat.completions.create(
            model=self._settings.llm.model,
            messages=messages,
        )
        text = final_response.choices[0].message.content or ""
        await self.conversation_store.add_message(device_id, MessageRole.ASSISTANT, text)
        return text

    async def _get_tools_schema(self) -> Optional[list[dict[str, Any]]]:
        from tools.registry import get_tool_registry
        registry = await get_tool_registry()
        tools = registry.get_all()
        return [self._tool_to_schema(t) for t in tools.values()]

    def _tool_to_schema(self, tool) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema,
            },
        }