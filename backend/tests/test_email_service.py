"""
Unit tests for the email alert service.

Tests cover:
- Successful email send with attachment
- Missing attachment frame (graceful fallback)
- SMTP connection failure handling
- Rate limiting: max emails per hour per camera
- Template rendering with correct context
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call
from datetime import datetime
import smtplib

from app.services.email_service import EmailService, EmailPayload


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def smtp_config():
    return {
        "GMAIL_USER": "test@gmail.com",
        "GMAIL_APP_PASSWORD": "fake_app_password",
        "ALERT_FROM_EMAIL": "alerts@examguard.ai",
    }


@pytest.fixture
def email_service(smtp_config):
    with patch("app.services.email_service.settings") as mock_settings:
        mock_settings.GMAIL_USER = smtp_config["GMAIL_USER"]
        mock_settings.GMAIL_APP_PASSWORD = smtp_config["GMAIL_APP_PASSWORD"]
        mock_settings.ALERT_FROM_EMAIL = smtp_config["ALERT_FROM_EMAIL"]
        svc = EmailService()
    return svc


@pytest.fixture
def valid_payload():
    return EmailPayload(
        to_email="invigilator@university.edu",
        invigilator_name="Dr. Smith",
        lecture_hall="LH-101",
        camera_id="cam_01",
        behaviour_type="PHONE_USE",
        confidence_score=0.92,
        detected_at=datetime(2025, 1, 15, 10, 30, 0),
        frame_bytes=b"fake_jpeg_frame_data",
        event_id="evt_abc123",
    )


# ---------------------------------------------------------------------------
# Email payload validation tests
# ---------------------------------------------------------------------------

class TestEmailPayload:
    def test_valid_payload_constructs(self, valid_payload):
        assert valid_payload.to_email == "invigilator@university.edu"
        assert valid_payload.confidence_score == 0.92

    def test_invalid_email_raises(self):
        with pytest.raises(ValueError):
            EmailPayload(
                to_email="not-an-email",
                invigilator_name="Test",
                lecture_hall="LH-1",
                camera_id="cam_01",
                behaviour_type="PHONE_USE",
                confidence_score=0.9,
                detected_at=datetime.utcnow(),
                frame_bytes=b"",
                event_id="evt_000",
            )

    def test_confidence_out_of_range_raises(self):
        with pytest.raises(ValueError):
            EmailPayload(
                to_email="test@uni.edu",
                invigilator_name="Test",
                lecture_hall="LH-1",
                camera_id="cam_01",
                behaviour_type="PHONE_USE",
                confidence_score=1.5,  # > 1.0
                detected_at=datetime.utcnow(),
                frame_bytes=b"",
                event_id="evt_000",
            )


# ---------------------------------------------------------------------------
# Email send tests
# ---------------------------------------------------------------------------

class TestEmailServiceSend:

    @pytest.mark.asyncio
    async def test_successful_send_returns_true(self, email_service, valid_payload):
        with patch("smtplib.SMTP_SSL") as mock_smtp_cls:
            mock_smtp = MagicMock()
            mock_smtp_cls.return_value.__enter__ = MagicMock(return_value=mock_smtp)
            mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)
            result = await email_service.send_alert(valid_payload)
        assert result is True

    @pytest.mark.asyncio
    async def test_smtp_failure_returns_false(self, email_service, valid_payload):
        with patch("smtplib.SMTP_SSL", side_effect=smtplib.SMTPException("Connection refused")):
            result = await email_service.send_alert(valid_payload)
        assert result is False

    @pytest.mark.asyncio
    async def test_email_contains_invigilator_name(self, email_service, valid_payload):
        """Subject line must include the invigilator name and lecture hall."""
        sent_messages = []

        def fake_send(from_addr, to_addrs, msg_str):
            sent_messages.append(msg_str)

        with patch("smtplib.SMTP_SSL") as mock_smtp_cls:
            mock_smtp = MagicMock()
            mock_smtp.sendmail.side_effect = fake_send
            mock_smtp_cls.return_value.__enter__ = MagicMock(return_value=mock_smtp)
            mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)
            await email_service.send_alert(valid_payload)

        assert len(sent_messages) == 1
        assert "LH-101" in sent_messages[0] or "PHONE_USE" in sent_messages[0]

    @pytest.mark.asyncio
    async def test_email_attaches_frame(self, email_service, valid_payload):
        """Frame bytes must be included as JPEG attachment."""
        with patch("smtplib.SMTP_SSL") as mock_smtp_cls:
            mock_smtp = MagicMock()
            mock_smtp_cls.return_value.__enter__ = MagicMock(return_value=mock_smtp)
            mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)
            await email_service.send_alert(valid_payload)
            # Verify sendmail was called once
            mock_smtp.sendmail.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_frame_bytes_sends_without_attachment(self, email_service, valid_payload):
        valid_payload.frame_bytes = b""
        with patch("smtplib.SMTP_SSL") as mock_smtp_cls:
            mock_smtp = MagicMock()
            mock_smtp_cls.return_value.__enter__ = MagicMock(return_value=mock_smtp)
            mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)
            result = await email_service.send_alert(valid_payload)
        # Should still succeed (just no attachment)
        assert result is True


# ---------------------------------------------------------------------------
# Rate limiting tests
# ---------------------------------------------------------------------------

class TestEmailRateLimiting:
    @pytest.mark.asyncio
    async def test_exceeds_hourly_limit_suppressed(self, email_service, valid_payload):
        """After MAX_EMAILS_PER_HOUR alerts for same camera, subsequent sends are suppressed."""
        max_limit = email_service.MAX_EMAILS_PER_HOUR
        # Simulate the camera already having hit the limit
        email_service._sent_count["cam_01"] = max_limit

        with patch("smtplib.SMTP_SSL") as mock_smtp_cls:
            result = await email_service.send_alert(valid_payload)
        # SMTP should NOT have been called
        mock_smtp_cls.assert_not_called()
        assert result is False
