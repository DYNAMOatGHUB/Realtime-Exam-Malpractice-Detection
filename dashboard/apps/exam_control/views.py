"""
Exam Control views — HEC operational interface.
Covers: LH Mapping, Live Monitor, Review Queue, Alert Log.
All views proxy to FastAPI for live data; Django provides auth + rendering.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime

import requests
from django.conf import settings
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.core.paginator import Paginator
from django.http import JsonResponse, StreamingHttpResponse
from django.shortcuts import render, redirect, get_object_or_404
from django.views.decorators.http import require_POST
from django_htmx.http import HttpResponseClientRefresh

from apps.accounts.decorators import hec_or_admin_required
from apps.exam_control.models import LectureHall, Invigilator, LHMapping, VideoAnalysisJob
from apps.exam_control.forms import LHMappingForm, InvigilatorForm, VideoAnalysisJobForm

logger = logging.getLogger(__name__)
FASTAPI = settings.FASTAPI_BASE_URL


def _api(path: str, method: str = "GET", **kwargs):
    """Helper: call FastAPI, return (data, ok)."""
    url = f"{FASTAPI}{path}"
    try:
        resp = getattr(requests, method.lower())(url, timeout=5, **kwargs)
        resp.raise_for_status()
        return resp.json(), True
    except requests.RequestException as exc:
        logger.error("FastAPI call %s %s failed: %s", method, url, exc)
        return {"error": str(exc)}, False


# ── Dashboard home ─────────────────────────────────────────────

@hec_or_admin_required
def dashboard(request):
    stats, ok = _api("/api/detections/stats")
    stream_data, _ = _api("/api/streams/active")
    active_streams = stream_data.get("active_streams", []) if _ else []

    return render(request, "exam_control/dashboard.html", {
        "stats": stats if ok else {},
        "active_streams": active_streams,
        "page_title": "Dashboard",
    })


# ── LH Mapping CRUD ───────────────────────────────────────────

@hec_or_admin_required
def lh_mapping_list(request):
    mappings = LHMapping.objects.filter(is_active=True).select_related(
        "lecture_hall", "invigilator"
    ).order_by("-created_at")
    return render(request, "exam_control/lh_mapping.html", {
        "mappings": mappings,
        "form": LHMappingForm(),
        "page_title": "LH–Invigilator Mapping",
    })


@hec_or_admin_required
@require_POST
def lh_mapping_create(request):
    form = LHMappingForm(request.POST)
    if form.is_valid():
        lh_code = form.cleaned_data["lh_code"]
        lh_name = form.cleaned_data["lh_name"]
        invigilator_name = form.cleaned_data["invigilator_name"]
        invigilator_email = form.cleaned_data["invigilator_email"]

        # get or create Lecture Hall
        lh, _ = LectureHall.objects.get_or_create(
            lh_code=lh_code,
            defaults={"name": lh_name}
        )
        if not _:
            # update name if it already existed
            lh.name = lh_name
            lh.save(update_fields=["name"])
            
        # get or create Invigilator
        invig, _ = Invigilator.objects.get_or_create(
            email=invigilator_email,
            defaults={"full_name": invigilator_name}
        )
        if not _:
            invig.full_name = invigilator_name
            invig.save(update_fields=["full_name"])

        mapping = form.save(commit=False)
        mapping.lecture_hall = lh
        mapping.invigilator = invig
        
        import uuid
        mapping.camera_id = f"batch_{uuid.uuid4().hex[:8]}"
        mapping.rtsp_url = "batch_upload"
        
        mapping.save()
        
        messages.success(
            request,
            f"Mapped {mapping.lecture_hall.lh_code} → {mapping.invigilator.full_name}"
        )
        if request.htmx:
            return HttpResponseClientRefresh()
        return redirect("exam_control:lh_mapping")
    return render(request, "exam_control/lh_mapping.html", {
        "form": form,
        "mappings": LHMapping.objects.filter(is_active=True).select_related("lecture_hall", "invigilator"),
        "page_title": "LH–Invigilator Mapping",
    })


@hec_or_admin_required
@require_POST
def lh_mapping_deactivate(request, mapping_id):
    mapping = get_object_or_404(LHMapping, id=mapping_id)
    mapping.is_active = False
    mapping.deactivated_at = datetime.utcnow()
    mapping.save(update_fields=["is_active", "deactivated_at"])
    messages.info(request, f"Mapping for {mapping.lecture_hall.lh_code} deactivated.")
    if request.htmx:
        return HttpResponseClientRefresh()
    return redirect("exam_control:lh_mapping")


# ── Video Analysis (Upload) ────────────────────────────────────

@hec_or_admin_required
def analyze_video(request):
    jobs = VideoAnalysisJob.objects.select_related("mapping__lecture_hall", "mapping__invigilator").all()
    form = VideoAnalysisJobForm()
    
    # only show mappings that are active
    form.fields['mapping'].queryset = LHMapping.objects.filter(is_active=True)
    
    return render(request, "exam_control/analyze_video.html", {
        "jobs": jobs,
        "form": form,
        "fastapi_url": FASTAPI,
        "page_title": "Analyze Video",
        "active_page": "analyze_video",
    })

@hec_or_admin_required
@require_POST
def upload_video(request):
    """
    Accepts the uploaded video, saves the job record, then immediately
    returns JSON so the browser can redirect. FastAPI is dispatched in a
    background thread so the browser never blocks on AI pipeline startup.
    """
    from django.http import JsonResponse
    import threading

    logger.info("Upload video view invoked.")
    form = VideoAnalysisJobForm(request.POST, request.FILES)
    if not form.is_valid():
        errors = {field: errs.as_text() for field, errs in form.errors.items()}
        logger.error("Upload form invalid: %s", errors)
        return JsonResponse({"success": False, "errors": errors}, status=400)

    logger.info("Form is valid. Saving job to DB...")
    job = form.save(commit=False)
    job.status = "processing"
    job.save()
    logger.info("Job %s saved. Starting background thread...", job.id)

    video_relative = job.video_file.name   # e.g. "cctv_uploads/filename.mp4"
    payload = {
        "job_id": str(job.id),
        "video_path": video_relative,
        "mapping_id": str(job.mapping.id) if job.mapping else None,
    }

    # Fire-and-forget: dispatch to FastAPI without blocking the HTTP response.
    # This means the browser gets its redirect instantly at 100% upload,
    # while the AI pipeline starts asynchronously in the background.
    def _dispatch_to_fastapi():
        try:
            import requests as req_lib
            req_lib.post(
                f"{FASTAPI}/api/video/analyze",
                json=payload,
                timeout=60,   # generous but not blocking the user
            )
        except Exception as exc:
            logger.warning("Background FastAPI dispatch error for job %s: %s", job.id, exc)

    threading.Thread(target=_dispatch_to_fastapi, daemon=True).start()

    # Immediately tell the browser the upload succeeded — redirect now.
    return JsonResponse({
        "success": True,
        "job_id": str(job.id),
        "redirect": "/analyze-video/",
    })




@hec_or_admin_required
def analyze_video_detail(request, job_id):
    """Render the real-time analytics page for a specific video job."""
    job = get_object_or_404(VideoAnalysisJob, id=job_id)
    return render(request, "exam_control/analyze_video_detail.html", {
        "job": job,
        "fastapi_url": FASTAPI,
        "page_title": "Live Analysis",
        "active_page": "analyze_video",
    })


@hec_or_admin_required
@require_POST
def delete_job(request, job_id):
    """
    Delete a video analysis job and its associated video file from disk.
    POST only — called via a small form with a CSRF token.
    """
    import os
    job = get_object_or_404(VideoAnalysisJob, id=job_id)
    filename = job.video_file.name if job.video_file else "unknown"

    # 1. Delete the physical video file from media storage
    try:
        if job.video_file and job.video_file.storage.exists(job.video_file.name):
            job.video_file.storage.delete(job.video_file.name)
    except Exception as exc:
        logger.warning("Could not delete video file %s: %s", filename, exc)

    # 2. Delete output / annotated video if present
    try:
        if job.output_video_file and job.output_video_file.storage.exists(job.output_video_file.name):
            job.output_video_file.storage.delete(job.output_video_file.name)
    except Exception as exc:
        logger.warning("Could not delete output file: %s", exc)

    # 3. Delete the DB record
    job.delete()

    short_name = filename.split("/")[-1] if "/" in filename else filename
    messages.success(request, f"Job and video \"{short_name}\" deleted successfully.")
    return redirect("exam_control:analyze_video")


@hec_or_admin_required
def video_stream_proxy(request, job_id):
    """
    SSE proxy: forwards the FastAPI SSE stream to the browser.
    Avoids CORS issues by letting Django relay the event stream.
    """
    import json as _json
    import urllib.parse

    job = get_object_or_404(VideoAnalysisJob, id=job_id)
    video_path = job.video_file.name  # relative path inside media/
    encoded_path = urllib.parse.quote(video_path, safe='')
    fastapi_stream_url = f"{FASTAPI}/api/video/stream/{job_id}?video_path={encoded_path}"

    def event_stream():
        try:
            with requests.get(fastapi_stream_url, stream=True, timeout=600) as r:
                # chunk_size=None lets requests return data as it arrives,
                # preserving SSE message boundaries for large JPEG payloads
                for chunk in r.iter_content(chunk_size=None):
                    if chunk:
                        yield chunk
        except Exception as exc:
            logger.error("SSE proxy error for job %s: %s", job_id, exc)
            yield f'data: {_json.dumps({"type": "error", "message": str(exc)})}\n\n'.encode()

    response = StreamingHttpResponse(
        event_stream(),
        content_type='text/event-stream',
    )
    response['Cache-Control'] = 'no-cache'
    response['X-Accel-Buffering'] = 'no'
    # response['Connection'] = 'keep-alive'
    return response



# ── Review Queue (Layer 8) ─────────────────────────────────────

@hec_or_admin_required
def review_queue(request):
    data, ok = _api("/api/alerts/review-queue?limit=50")
    items = data.get("items", []) if ok else []
    stats, _ = _api("/api/detections/stats")
    return render(request, "exam_control/review_queue.html", {
        "queue_items": items,
        "queue_length": len(items),
        "stats": stats if _ else {},
        "page_title": "Review Queue",
    })


@hec_or_admin_required
@require_POST
def review_action(request):
    event_id = request.POST.get("event_id")
    action = request.POST.get("action")
    note = request.POST.get("note", "")

    if not event_id or action not in ("confirm", "dismiss"):
        messages.error(request, "Invalid review action.")
        return redirect("exam_control:review_queue")

    payload = {
        "event_id": event_id,
        "reviewer_id": str(request.user.id),
        "action": action,
        "note": note,
    }
    data, ok = _api("/api/alerts/review", method="POST", json=payload)
    if ok:
        status_label = "CONFIRMED" if action == "confirm" else "DISMISSED"
        messages.success(request, f"Event {event_id[:8]}… marked as {status_label}")
    else:
        messages.error(request, f"Review failed: {data.get('detail', 'Unknown error')}")

    if request.htmx:
        return HttpResponseClientRefresh()
    return redirect("exam_control:review_queue")


# ── Alert Log ─────────────────────────────────────────────────

@hec_or_admin_required
def alert_log(request):
    page = int(request.GET.get("page", 1))
    data, ok = _api(f"/api/alerts/log?page={page}&page_size=20")
    return render(request, "exam_control/alert_log.html", {
        "alerts": data.get("results", []) if ok else [],
        "total": data.get("total", 0) if ok else 0,
        "page": page,
        "page_title": "Alert Log",
    })


# ── HTMX partial: stats refresh ───────────────────────────────

@hec_or_admin_required
def stats_partial(request):
    stats, ok = _api("/api/detections/stats")
    return render(request, "exam_control/partials/stats_cards.html", {
        "stats": stats if ok else {},
    })
