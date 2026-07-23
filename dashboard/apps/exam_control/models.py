"""
Django ORM models for exam_control (mirrors FastAPI/PostgreSQL schema).
These are used for Django's ORM-backed forms and admin display.
"""
from django.db import models
import uuid


class Invigilator(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    full_name = models.CharField(max_length=255)
    email = models.EmailField(unique=True, db_index=True)
    department = models.CharField(max_length=255, blank=True, null=True)
    phone = models.CharField(max_length=20, blank=True, null=True)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["full_name"]
        verbose_name = "Invigilator"
        verbose_name_plural = "Invigilators"

    def __str__(self):
        return f"{self.full_name} ({self.email})"


class LectureHall(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    lh_code = models.CharField(max_length=50, unique=True)
    name = models.CharField(max_length=255)
    building = models.CharField(max_length=255, blank=True, null=True)
    capacity = models.IntegerField(null=True, blank=True)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["lh_code"]
        verbose_name = "Lecture Hall"
        verbose_name_plural = "Lecture Halls"

    def __str__(self):
        return f"{self.lh_code} — {self.name}"


class LHMapping(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    lecture_hall = models.ForeignKey(LectureHall, on_delete=models.PROTECT, related_name="mappings")
    invigilator = models.ForeignKey(Invigilator, on_delete=models.PROTECT, related_name="mappings")
    camera_id = models.CharField(max_length=100, db_index=True)
    rtsp_url = models.CharField(max_length=512)
    exam_session_label = models.CharField(max_length=255, blank=True, null=True)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    deactivated_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "LH–Invigilator Mapping"
        verbose_name_plural = "LH–Invigilator Mappings"

    def __str__(self):
        return f"{self.lecture_hall.lh_code} → {self.invigilator.full_name} (cam: {self.camera_id})"


class VideoAnalysisJob(models.Model):
    STATUS_CHOICES = [
        ("pending", "Pending"),
        ("processing", "Processing"),
        ("completed", "Completed"),
        ("failed", "Failed"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    mapping = models.ForeignKey(LHMapping, on_delete=models.SET_NULL, null=True, blank=True, related_name="video_jobs")
    video_file = models.FileField(upload_to="cctv_uploads/")
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="pending")
    total_anomalies = models.IntegerField(default=0)
    output_video_file = models.FileField(upload_to="processed_videos/", null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "Video Analysis Job"
        verbose_name_plural = "Video Analysis Jobs"

    def __str__(self):
        return f"Job {self.id} ({self.get_status_display()})"
