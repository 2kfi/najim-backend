from datetime import datetime
from typing import Optional
from core.redis_manager import RedisManager
from core.schemas import DeviceInfo, DeviceStatus
from core.config import get_settings


class DeviceRegistry:
    KEY = "devices"
    WS_NODE_PREFIX = "device_ws"

    def __init__(self, redis: RedisManager):
        self.redis = redis
        self._settings = get_settings()

    def _ws_key(self, device_id: str) -> str:
        return f"{self.WS_NODE_PREFIX}:{device_id}"

    async def register(self, device_id: str, info: DeviceInfo) -> None:
        now = datetime.utcnow()
        data = {
            "device_id": device_id,
            "user_id": info.user_id or "",
            "host": info.host or "",
            "port": str(info.port or ""),
            "capabilities": info.capabilities if isinstance(info.capabilities, str) else ",".join(info.capabilities),
            "status": info.status.value,
            "connected_at": now.isoformat(),
            "last_heartbeat": now.isoformat(),
            "node_id": self._settings.cluster.node_id,
        }
        await self.redis.hset_dict(self.KEY, {device_id: data})

    async def unregister(self, device_id: str) -> None:
        client = self.redis.client
        await client.hdel(self.KEY, device_id)
        await self.redis.delete(self._ws_key(device_id))

    async def get(self, device_id: str) -> Optional[DeviceInfo]:
        raw = await self.redis.hget_all(self.KEY)
        if not raw:
            return None
        device_data = raw.get(device_id)
        if not device_data:
            return None
        if isinstance(device_data, dict):
            return DeviceInfo(
                device_id=device_data.get("device_id", device_id),
                user_id=device_data.get("user_id") or None,
                host=device_data.get("host") or None,
                port=int(device_data["port"]) if device_data.get("port") else None,
                capabilities=device_data.get("capabilities", "").split(",") if device_data.get("capabilities") else [],
                status=DeviceStatus(device_data.get("status", "online")) if device_data.get("status") in ["online", "offline", "busy"] else DeviceStatus.ONLINE,
                connected_at=datetime.fromisoformat(device_data["connected_at"]) if device_data.get("connected_at") else datetime.utcnow(),
                last_heartbeat=datetime.fromisoformat(device_data["last_heartbeat"]) if device_data.get("last_heartbeat") else datetime.utcnow(),
                node_id=device_data.get("node_id"),
            )
        return None

    async def get_node_for_device(self, device_id: str) -> Optional[str]:
        return await self.redis.get(self._ws_key(device_id))

    async def heartbeat(self, device_id: str) -> None:
        now = datetime.utcnow().isoformat()
        all_devices = await self.redis.hget_all(self.KEY)
        if device_id in all_devices:
            data = all_devices[device_id]
            if isinstance(data, dict):
                await self.redis.hset_dict(self.KEY, {device_id: {**data, "last_heartbeat": now}})

    async def set_status(self, device_id: str, status: DeviceStatus) -> None:
        all_devices = await self.redis.hget_all(self.KEY)
        if device_id in all_devices:
            data = all_devices[device_id]
            if isinstance(data, dict):
                await self.redis.hset_dict(self.KEY, {device_id: {**data, "status": status.value}})

    async def is_online(self, device_id: str) -> bool:
        info = await self.get(device_id)
        return info is not None and info.status == DeviceStatus.ONLINE

    async def list_connected(self) -> list[DeviceInfo]:
        raw = await self.redis.hget_all(self.KEY)
        devices = []
        for device_id, data in raw.items():
            dev_id = device_id if isinstance(device_id, str) else device_id.decode()
            if isinstance(data, dict):
                devices.append(DeviceInfo(
                    device_id=dev_id,
                    user_id=data.get("user_id") or None,
                    host=data.get("host") or None,
                    port=int(data["port"]) if data.get("port") else None,
                    capabilities=data.get("capabilities", "").split(",") if data.get("capabilities") else [],
                    status=DeviceStatus(data.get("status", "online")) if data.get("status") in ["online", "offline", "busy"] else DeviceStatus.ONLINE,
                    connected_at=datetime.fromisoformat(data["connected_at"]) if data.get("connected_at") else datetime.utcnow(),
                    last_heartbeat=datetime.fromisoformat(data["last_heartbeat"]) if data.get("last_heartbeat") else datetime.utcnow(),
                    node_id=data.get("node_id"),
                ))
        return devices

    async def count(self) -> int:
        return await self.redis.client.hlen(self.KEY)