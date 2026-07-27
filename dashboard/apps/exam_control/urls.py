"""URL configuration for exam_control app."""
from django.urls import path
from apps.exam_control import views

app_name = "exam_control"

urlpatterns = [
    # Dashboard
    path("", views.dashboard, name="dashboard"),
    path("stats-partial/", views.stats_partial, name="stats_partial"),

    # LH Mapping
    path("lh-mapping/", views.lh_mapping_list, name="lh_mapping"),
    path("lh-mapping/create/", views.lh_mapping_create, name="lh_mapping_create"),
    path("lh-mapping/<uuid:mapping_id>/deactivate/", views.lh_mapping_deactivate, name="lh_mapping_deactivate"),

    # Analyze Video
    path("analyze-video/", views.analyze_video, name="analyze_video"),
    path("analyze-video/upload/", views.upload_video, name="upload_video"),
    path("analyze-video/<uuid:job_id>/", views.analyze_video_detail, name="analyze_video_detail"),
    path("analyze-video/<uuid:job_id>/stream/", views.video_stream_proxy, name="video_stream_proxy"),
    path("analyze-video/<uuid:job_id>/delete/", views.delete_job, name="delete_job"),
    path("analyze-video/<uuid:job_id>/complete/", views.update_job_status, name="update_job_status"),
    path("analyze-video/<uuid:job_id>/alert/", views.alert_invigilator, name="alert_invigilator"),

    # Review Queue (Layer 8)
    path("review-queue/", views.review_queue, name="review_queue"),
    path("review-action/", views.review_action, name="review_action"),

    # Alert Log
    path("alert-log/", views.alert_log, name="alert_log"),
]
