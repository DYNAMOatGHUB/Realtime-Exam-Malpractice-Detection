"""Django URL configuration."""
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path("django-admin/", admin.site.urls),
    path("accounts/", include("apps.accounts.urls")),
    path("", include("apps.exam_control.urls")),
    path("exam-control/", include("apps.exam_control.urls")),  # Fallback for cached JS
    path("admin-panel/", include("apps.admin_panel.urls")),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
