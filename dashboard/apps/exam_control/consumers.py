"""
Django Channels WebSocket consumer for live camera feed events.
Subscribes to Redis pub/sub channel for a specific camera_id
and forwards detection events to connected browser clients.
"""
import json
import logging
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.layers import get_channel_layer

logger = logging.getLogger(__name__)


class LiveFeedConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.camera_id = self.scope["url_route"]["kwargs"]["camera_id"]
        self.group_name = f"camera_{self.camera_id}"

        await self.channel_layer.group_add(self.group_name, self.channel_name)
        await self.accept()
        logger.info("WebSocket connected: camera=%s", self.camera_id)

    async def disconnect(self, close_code):
        await self.channel_layer.group_discard(self.group_name, self.channel_name)
        logger.info("WebSocket disconnected: camera=%s", self.camera_id)

    async def receive(self, text_data=None, bytes_data=None):
        """Client messages (e.g., ping) — not used in this version."""
        pass

    async def detection_event(self, event):
        """Forward a detection event message to the WebSocket client."""
        await self.send(text_data=json.dumps(event["data"]))
