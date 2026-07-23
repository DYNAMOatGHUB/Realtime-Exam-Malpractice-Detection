"""WebSocket URL routing for exam_control app (Django Channels)."""
from django.urls import re_path
from apps.exam_control import consumers

websocket_urlpatterns = [
    re_path(r"ws/live/(?P<camera_id>[^/]+)/$", consumers.LiveFeedConsumer.as_asgi()),
]
