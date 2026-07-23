from django.urls import path
from . import views

app_name = "admin_panel"

urlpatterns = [
    path("", views.dashboard_home, name="home"),
    path("users/", views.user_list, name="user_list"),
    path("users/create/", views.user_create, name="user_create"),
    path("users/<int:pk>/edit/", views.user_edit, name="user_edit"),
    path("users/<int:pk>/delete/", views.user_delete, name="user_delete"),
    path("users/<int:pk>/reset-password/", views.user_reset_password, name="user_reset_password"),
    path("models/", views.model_list, name="model_list"),
    path("models/upload/", views.model_upload, name="model_upload"),
    path("health/", views.system_health, name="system_health"),
    path("health/api/", views.health_api, name="health_api"),
]
