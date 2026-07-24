"""
Email alert service — Layer 7.
Sends malpractice alert emails via Gmail SMTP (free tier).
Attaches the annotated evidence frame as a JPEG attachment.
Respects daily send limit to stay within Gmail free-tier constraints.
"""
from __future__ import annotations

import email.mime.image
import logging
import smtplib
from datetime import date, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.base import MIMEBase
from email import encoders
from typing import Any

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Redis key for daily email counter
DAILY_COUNT_KEY = "eids:email_sent:{date}"


def _daily_count_key() -> str:
    return DAILY_COUNT_KEY.format(date=date.today().isoformat())


def _check_and_increment_daily_count() -> bool:
    """
    Atomically check and increment today's email send count.
    Returns True if under limit, False if limit reached.
    """
    r = get_sync_redis()
    key = _daily_count_key()
    count = r.incr(key)
    if count == 1:
        # Set TTL to 25h (persists through midnight, then auto-expires)
        r.expire(key, 90000)

    if count > settings.daily_email_limit:
        logger.warning(
            "Daily email limit (%d) reached — suppressing alert",
            settings.daily_email_limit,
        )
        return False
    return True


def _build_html_body(
    invigilator_name: str,
    lecture_hall: str,
    malpractice_class: str,
    confidence: float,
    timestamp: str,
    camera_id: str,
    presigned_url: str,
    event_id: str,
) -> str:
    class_label = malpractice_class.replace("_", " ").title()
    conf_pct = f"{confidence:.1%}"

    return f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <style>
    body {{ font-family: Arial, sans-serif; background:#f5f5f5; margin:0; padding:20px; }}
    .card {{ background:#fff; border-radius:8px; padding:24px; max-width:600px;
             margin:0 auto; box-shadow:0 2px 8px rgba(0,0,0,0.12); }}
    .header {{ background:#c0392b; color:#fff; padding:16px 24px; border-radius:6px 6px 0 0;
               margin:-24px -24px 20px; }}
    .badge {{ display:inline-block; background:#e74c3c; color:#fff; padding:4px 12px;
              border-radius:4px; font-weight:bold; font-size:14px; }}
    .meta {{ background:#f8f8f8; border-radius:4px; padding:12px; margin:16px 0; }}
    .meta td {{ padding:4px 12px 4px 0; font-size:14px; }}
    .meta .label {{ color:#666; font-weight:bold; white-space:nowrap; }}
    .action {{ background:#2980b9; color:#fff; text-decoration:none; padding:10px 20px;
               border-radius:4px; display:inline-block; margin-top:16px; }}
    .footer {{ font-size:12px; color:#999; margin-top:20px; }}
  </style>
</head>
<body>
  <div class="card">
    <div class="header">
      <h2 style="margin:0">⚠️ Malpractice Alert — Exam Vigilance System</h2>
    </div>

    <p>Dear <strong>{invigilator_name}</strong>,</p>
    <p>
      The automated monitoring system has detected a high-confidence malpractice event
      in your assigned exam hall. <strong>Please review and take appropriate action immediately.</strong>
    </p>

    <div class="badge">{class_label}</div>

    <div class="meta">
      <table>
        <tr><td class="label">Lecture Hall:</td><td>{lecture_hall}</td></tr>
        <tr><td class="label">Detected At:</td><td>{timestamp}</td></tr>
        <tr><td class="label">Camera ID:</td><td>{camera_id}</td></tr>
        <tr><td class="label">Confidence:</td><td>{conf_pct}</td></tr>
        <tr><td class="label">Event ID:</td><td><code>{event_id}</code></td></tr>
      </table>
    </div>

    <p>An annotated evidence frame is attached to this email.
       You can also view the full evidence (valid 24h) at:</p>
    <a class="action" href="{presigned_url}">View Evidence Frame →</a>

    <p style="margin-top:20px; color:#666; font-size:13px;">
      <em>This alert was generated automatically. All events are also visible in the
      Head of Exam Cell dashboard for human review before any disciplinary action.</em>
    </p>

    <div class="footer">
      Exam Vigilance System &bull; Self-hosted &bull; Event {event_id}
    </div>
  </div>
</body>
</html>
"""


def send_malpractice_alert(
    recipient_email: str,
    recipient_name: str,
    lecture_hall_name: str,
    malpractice_class: str,
    confidence_score: float,
    timestamp: str,
    camera_id: str,
    presigned_url: str,
    event_id: str,
    frame_bytes: bytes | None = None,
) -> dict[str, Any]:
    """
    Send an alert email to the mapped invigilator.

    Returns:
        dict with 'success', 'suppressed' (daily limit), 'error' fields.
    """
    if not _check_and_increment_daily_count():
        return {"success": False, "suppressed": True, "error": "daily limit reached"}

    subject = (
        f"[ALERT] Malpractice Detected — {lecture_hall_name} — "
        f"{malpractice_class.replace('_', ' ').title()}"
    )

    html_body = _build_html_body(
        invigilator_name=recipient_name,
        lecture_hall=lecture_hall_name,
        malpractice_class=malpractice_class,
        confidence=confidence_score,
        timestamp=timestamp,
        camera_id=camera_id,
        presigned_url=presigned_url,
        event_id=event_id,
    )

    msg = MIMEMultipart("related")
    msg["From"] = f"{settings.smtp_from_name} <{settings.smtp_user}>"
    msg["To"] = recipient_email
    msg["Subject"] = subject

    # HTML body
    msg.attach(MIMEText(html_body, "html"))

    # Attach evidence frame
    if frame_bytes:
        img_part = MIMEImage(frame_bytes, "jpeg")
        img_part.add_header(
            "Content-Disposition",
            "attachment",
            filename=f"evidence_{event_id}.jpg",
        )
        img_part.add_header("Content-ID", "<evidence_frame>")
        msg.attach(img_part)

    try:
        with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=15) as smtp:
            smtp.ehlo()
            smtp.starttls()
            smtp.login(settings.smtp_user, settings.smtp_password)
            smtp.sendmail(settings.smtp_user, [recipient_email], msg.as_string())

        logger.info(
            "Alert email sent to %s for event %s (class=%s conf=%.2f)",
            recipient_email, event_id, malpractice_class, confidence_score,
        )
        return {"success": True, "suppressed": False, "error": None}

    except smtplib.SMTPAuthenticationError as exc:
        logger.error("SMTP authentication failed: %s", exc)
        return {"success": False, "suppressed": False, "error": f"SMTP auth error: {exc}"}
    except smtplib.SMTPException as exc:
        logger.error("SMTP send error: %s", exc)
        return {"success": False, "suppressed": False, "error": str(exc)}
    except Exception as exc:
        logger.error("Unexpected email error: %s", exc)
        return {"success": False, "suppressed": False, "error": str(exc)}
