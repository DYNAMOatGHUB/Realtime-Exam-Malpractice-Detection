"""
Unit tests for the confidence-triage alert router.

Tests cover:
- HIGH confidence  → auto-alert + evidence packaging
- MID confidence   → review queue
- LOW confidence   → discard
- Edge cases: boundary values, invalid inputs
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from app.services.alert_router import AlertRouter, ConfidenceLevel, TriageResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_db():
    return AsyncMock()


@pytest.fixture
def mock_minio():
    return MagicMock()


@pytest.fixture
def router(mock_db, mock_minio):
    return AlertRouter(db=mock_db, minio_client=mock_minio)


def _make_frame_data(confidence: float, camera_id: str = "cam_01") -> dict:
    return {
        "camera_id": camera_id,
        "frame": b"fake_jpeg_bytes",
        "timestamp": datetime.utcnow().isoformat(),
        "confidence_score": confidence,
        "behaviour_type": "PHONE_USE",
        "bbox": [100, 100, 300, 400],
        "pose_keypoints": [[0.5, 0.5, 0.9]] * 17,
    }


# ---------------------------------------------------------------------------
# ConfidenceLevel classification tests
# ---------------------------------------------------------------------------

class TestConfidenceLevel:
    def test_high_confidence_threshold(self):
        assert ConfidenceLevel.classify(0.85) == ConfidenceLevel.HIGH

    def test_mid_confidence_lower_bound(self):
        assert ConfidenceLevel.classify(0.5) == ConfidenceLevel.MID

    def test_mid_confidence_upper_bound(self):
        assert ConfidenceLevel.classify(0.79) == ConfidenceLevel.MID

    def test_low_confidence(self):
        assert ConfidenceLevel.classify(0.3) == ConfidenceLevel.LOW

    def test_boundary_high(self):
        assert ConfidenceLevel.classify(0.80) == ConfidenceLevel.HIGH

    def test_boundary_mid(self):
        assert ConfidenceLevel.classify(0.499) == ConfidenceLevel.LOW

    def test_zero_confidence_is_low(self):
        assert ConfidenceLevel.classify(0.0) == ConfidenceLevel.LOW

    def test_perfect_confidence_is_high(self):
        assert ConfidenceLevel.classify(1.0) == ConfidenceLevel.HIGH


# ---------------------------------------------------------------------------
# AlertRouter triage tests
# ---------------------------------------------------------------------------

class TestAlertRouterTriage:

    @pytest.mark.asyncio
    async def test_high_confidence_triggers_auto_alert(self, router):
        frame = _make_frame_data(0.92)
        with patch.object(router, "_handle_high", new_callable=AsyncMock) as mock_high:
            mock_high.return_value = TriageResult(
                action="AUTO_ALERT",
                event_id="evt_001",
                confidence_level=ConfidenceLevel.HIGH,
            )
            result = await router.triage(frame)
        assert result.action == "AUTO_ALERT"
        mock_high.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_mid_confidence_goes_to_review_queue(self, router):
        frame = _make_frame_data(0.65)
        with patch.object(router, "_handle_mid", new_callable=AsyncMock) as mock_mid:
            mock_mid.return_value = TriageResult(
                action="REVIEW_QUEUE",
                event_id="evt_002",
                confidence_level=ConfidenceLevel.MID,
            )
            result = await router.triage(frame)
        assert result.action == "REVIEW_QUEUE"
        mock_mid.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_low_confidence_discards(self, router):
        frame = _make_frame_data(0.2)
        with patch.object(router, "_handle_low", new_callable=AsyncMock) as mock_low:
            mock_low.return_value = TriageResult(
                action="DISCARD",
                event_id=None,
                confidence_level=ConfidenceLevel.LOW,
            )
            result = await router.triage(frame)
        assert result.action == "DISCARD"
        assert result.event_id is None

    @pytest.mark.asyncio
    async def test_triage_missing_confidence_raises(self, router):
        frame = {"camera_id": "cam_01", "frame": b"data"}  # no confidence_score
        with pytest.raises((KeyError, ValueError)):
            await router.triage(frame)

    @pytest.mark.asyncio
    async def test_high_confidence_packages_evidence(self, router, mock_minio):
        """HIGH events must save a frame to MinIO."""
        frame = _make_frame_data(0.95)
        with patch("app.services.alert_router.evidence_packager") as mock_ep, \
             patch("app.services.alert_router.email_service") as mock_es:
            mock_ep.package = AsyncMock(return_value="s3://evidence-frames/evt_003.jpg")
            mock_es.send_alert = AsyncMock(return_value=True)
            result = await router._handle_high(frame)
        mock_ep.package.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_rate_limiting_suppresses_repeat_alerts(self, router):
        """Same camera should not trigger two auto-alerts within the cooldown window."""
        frame = _make_frame_data(0.95, camera_id="cam_02")
        router._last_alert_time["cam_02"] = datetime.utcnow()  # simulate recent alert
        with patch.object(router, "_handle_high", new_callable=AsyncMock) as mock_high:
            result = await router.triage(frame)
        # Should be throttled to REVIEW_QUEUE or DISCARD, not AUTO_ALERT
        assert result.action in ("REVIEW_QUEUE", "DISCARD", "THROTTLED")
